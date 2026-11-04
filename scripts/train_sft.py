#!/usr/bin/env python
"""LoRA SFT for Gemma 4 on the bf16 dequantization of the served w4a16 checkpoint.

E4B on 4xL4 (DDP, one sequence per GPU per micro-step):
    torchrun --nproc_per_node 4 scripts/train_sft.py --model-dir /kaggle/tmp/e4b-bf16 \
        --data data/sft_e4b --out /kaggle/working/adapters/main_lora --rank 32 --alpha 32 \
        --lr 1e-4 --epochs 1 --grad-accum 4 --ple-cpu

31B / 12B (single process, layers spread over the 4 GPUs; slow, short context):
    python scripts/train_sft.py --strategy device_map --model-dir ... --max-len 8192 ...

Design (report section "Train on dequantized w4a16 weights"):
  * LoRA on language-model q/k/v/o/gate/up/down only; never vision/audio, embeddings, lm_head;
    k/v_proj skipped on KV-shared layers (vLLM drops those modules).
  * default attn implementation (FA2 crashes on head_dim 512; explicit sdpa fails on the audio tower)
  * chunked + checkpointed LM head with Gemma's final-logit softcap (262k vocab)
  * loss normalised by the global number of label tokens per optimizer step
  * bf16 only (fp16 overflows on L4/T4), gradient checkpointing (non-reentrant)
  * aborts on NaN/inf loss; saves a PEFT dir (adapter_config.json + adapter_model.safetensors)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.losses import chunked_lm_loss, shift_labels  # noqa: E402

PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
TOWER_NAMES = ("vision_tower", "audio_tower", "embed_vision", "embed_audio", "multi_modal_projector")


# ----------------------------------------------------------------------------- data
class SFTData:
    def __init__(self, d: str | Path):
        d = Path(d)
        self.ids = np.load(d / "input_ids.npy", mmap_mode="r")
        self.mask = np.load(d / "loss_mask.npy", mmap_mode="r")
        self.index = json.loads((d / "index.json").read_text())

    def __len__(self):
        return len(self.index)

    def get(self, i: int, max_len: int):
        e = self.index[i]
        o, n = e["offset"], min(e["length"], max_len)
        ids = torch.from_numpy(np.asarray(self.ids[o : o + n], dtype=np.int64))[None]
        m = torch.from_numpy(np.asarray(self.mask[o : o + n], dtype=np.int64))[None]
        return ids, m


# ----------------------------------------------------------------------------- model helpers
class _CPUEmbedding(nn.Module):
    """Keeps a huge frozen embedding (E4B per-layer embeddings, ~5.6 GB) on CPU, invisible to DDP."""

    def __init__(self, inner: nn.Module, out_device):
        super().__init__()
        object.__setattr__(self, "_inner", inner.to("cpu"))
        self.out_device = out_device

    def forward(self, ids, *a, **k):
        return self._inner(ids.to("cpu"), *a, **k).to(self.out_device, non_blocking=True)


def _set_module(root: nn.Module, name: str, new: nn.Module):
    parent = root
    parts = name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def load_model(model_dir: str, strategy: str, local_rank: int, max_mem_gib: float):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_dir)
    kw = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if strategy == "ddp":
        kw["device_map"] = {"": "cpu"}  # load on CPU, move after surgery
    else:
        n = torch.cuda.device_count()
        kw["device_map"] = "auto"
        kw["max_memory"] = {i: f"{max_mem_gib}GiB" for i in range(n)} | {"cpu": "200GiB"}
    last_err = None
    archs = getattr(cfg, "architectures", None) or []
    loaders = []
    try:
        from transformers import AutoModelForImageTextToText

        if any("ConditionalGeneration" in a for a in archs):
            loaders.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    from transformers import AutoModelForCausalLM

    loaders.append(AutoModelForCausalLM)
    for L in loaders:
        try:
            return L.from_pretrained(model_dir, **kw)
        except Exception as e:  # try the next class
            last_err = e
    raise RuntimeError(f"could not load {model_dir}: {last_err}")


def drop_towers(model: nn.Module) -> list[str]:
    dropped = []
    for name, mod in list(model.named_modules()):
        leaf = name.split(".")[-1]
        if leaf in TOWER_NAMES and not any(name.startswith(d + ".") for d in dropped):
            try:
                _set_module(model, name, nn.Identity())
                dropped.append(name)
            except Exception:
                pass
    return dropped


def ple_to_cpu(model: nn.Module, device) -> list[str]:
    moved = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Embedding) and "per_layer" in name and mod.weight.numel() > float(os.environ.get("GSWE_PLE_MIN_NUMEL", 1e8)):
            mod.weight.requires_grad_(False)
            _set_module(model, name, _CPUEmbedding(mod, device))
            moved.append(name)
    return moved


def lora_targets(model: nn.Module, projs: tuple[str, ...]) -> tuple[list[str], dict]:
    tcfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers = int(getattr(tcfg, "num_hidden_layers", 0))
    n_shared = int(getattr(tcfg, "num_kv_shared_layers", 0) or 0)
    shared = set(range(n_layers - n_shared, n_layers)) if n_shared else set()
    targets, skipped = [], 0
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(t in name for t in ("vision", "audio", "embed", "lm_head")):
            continue
        m = LAYER_RE.search(name)
        if not m or m.group(3) not in projs:
            continue
        if int(m.group(1)) in shared and m.group(3) in ("k_proj", "v_proj"):
            skipped += 1
            continue
        targets.append(name)
    return targets, {"n_layers": n_layers, "kv_shared": sorted(shared), "skipped_kv_shared": skipped}


class LossModule(nn.Module):
    """forward(input_ids, labels) -> (sum_loss, n_tokens). Wrapped by DDP so grads sync."""

    def __init__(self, peft_model, chunk: int):
        super().__init__()
        self.m = peft_model
        inner = peft_model.get_base_model()
        self.decoder = getattr(inner, "model", None)
        self.lm_head = inner.get_output_embeddings()
        tcfg = inner.config.get_text_config() if hasattr(inner.config, "get_text_config") else inner.config
        self.softcap = getattr(tcfg, "final_logit_softcapping", None)
        self.chunk = chunk

    def forward(self, input_ids, labels):
        out = self.decoder(input_ids=input_ids, use_cache=False)
        h = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        w = self.lm_head.weight
        h = h.to(w.device)
        return chunked_lm_loss(h, w, labels.to(w.device), self.softcap, self.chunk)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strategy", choices=["ddp", "device_map"], default="ddp")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--projs", default=",".join(PROJ))
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--grad-accum", type=int, default=4, help="micro-steps per optimizer step, per rank")
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N optimizer steps (smoke tests)")
    ap.add_argument("--save-steps", type=int, default=0)
    ap.add_argument("--log-steps", type=int, default=5)
    ap.add_argument("--init-adapter", default=None, help="continue training an existing PEFT adapter")
    ap.add_argument("--ple-cpu", action="store_true", help="E4B/E2B: keep per-layer embedding table on CPU")
    ap.add_argument("--keep-towers", action="store_true")
    ap.add_argument("--max-mem-gib", type=float, default=21.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    distributed = a.strategy == "ddp" and int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local_rank = 0, 1, 0
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)

    from peft import LoraConfig, PeftModel, get_peft_model

    t0 = time.time()
    model = load_model(a.model_dir, a.strategy, local_rank, a.max_mem_gib)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    dropped = [] if a.keep_towers else drop_towers(model)
    moved = ple_to_cpu(model, device) if (a.ple_cpu and a.strategy == "ddp") else []
    if a.strategy == "ddp":
        model.to(device)
    log(f"loaded in {time.time() - t0:.0f}s; dropped towers={dropped}; PLE on CPU={moved}")

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
        info = {"init_adapter": a.init_adapter}
    else:
        targets, info = lora_targets(model, tuple(a.projs.split(",")))
        if not targets:
            raise SystemExit("no LoRA target modules found; inspect model.named_modules()")
        cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none",
                         target_modules=targets, task_type="CAUSAL_LM", init_lora_weights=True)
        model = get_peft_model(model, cfg)
        info["n_targets"] = len(targets)
        info["example_targets"] = targets[:3]
    if rank == 0:
        model.print_trainable_parameters()
    log(json.dumps(info))

    core = LossModule(model, a.chunk)
    if a.strategy == "ddp":
        core.to(device)
    net = nn.parallel.DistributedDataParallel(core, device_ids=[local_rank], broadcast_buffers=False,
                                              find_unused_parameters=False) if distributed else core

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.999), weight_decay=a.weight_decay)

    data = SFTData(a.data)
    n_epochs_ceil = math.ceil(a.epochs)
    order = []
    for ep in range(n_epochs_ceil):
        idx = list(range(len(data)))
        if ep > 0:
            rng = np.random.default_rng(a.seed + ep)
            rng.shuffle(idx)  # curriculum only on the first pass
        order += idx
    order = order[: int(len(data) * a.epochs)]
    per_rank = len(order) // world
    mine = order[rank::world][:per_rank]
    steps = per_rank // a.grad_accum
    if a.max_steps:
        steps = min(steps, a.max_steps)
    if steps == 0:
        raise SystemExit(f"not enough samples ({len(data)}) for world={world} x grad_accum={a.grad_accum}")
    warm = max(1, int(steps * a.warmup))

    def lr_at(s):
        if s < warm:
            return a.lr * (s + 1) / warm
        p = (s - warm) / max(1, steps - warm)
        return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    log(f"samples={len(data)} world={world} steps={steps} grad_accum={a.grad_accum} warmup={warm}")
    out = Path(a.out)
    tok_seen, t_start, ema = 0, time.time(), None
    net.train()
    for step in range(steps):
        batch_ids = mine[step * a.grad_accum : (step + 1) * a.grad_accum]
        batch = [data.get(i, a.max_len) for i in batch_ids]
        labels = [shift_labels(ids, m) for ids, m in batch]
        n_local = sum(int((lab != -100).sum()) for lab in labels)
        n_glob = torch.tensor([n_local], device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(n_glob)
        n_glob = max(1.0, float(n_glob.item()))
        tot = 0.0
        for j, ((ids, _), lab) in enumerate(zip(batch, labels)):
            sync = (j == len(batch) - 1) or not distributed
            ctx = nullcontext() if sync else net.no_sync()
            with ctx:
                loss_sum, n = net(ids.to(device), lab.to(device))
                loss = loss_sum * (world / n_glob)
                loss.backward()
            tot += float(loss_sum.detach())
            tok_seen += ids.shape[1]
        if not math.isfinite(tot):
            raise SystemExit(f"non-finite loss at step {step}; check attention impl / softcap / dtype")
        gn = torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step_loss = tot / max(1, n_local)
        ema = step_loss if ema is None else 0.9 * ema + 0.1 * step_loss
        if step % a.log_steps == 0 or step == steps - 1:
            el = time.time() - t_start
            mem = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0
            log(f"step {step + 1}/{steps} loss {step_loss:.4f} ema {ema:.4f} gnorm {float(gn):.3f} "
                f"lr {lr_at(step):.2e} tok/s(rank) {tok_seen / el:.0f} peak_mem {mem:.1f}GiB eta {(steps - step - 1) * el / (step + 1) / 60:.0f}min")
        if a.save_steps and rank == 0 and (step + 1) % a.save_steps == 0:
            model.save_pretrained(out / f"checkpoint-{step + 1}", safe_serialization=True)
    if rank == 0:
        model.save_pretrained(out, safe_serialization=True)
        (out / "train_args.json").write_text(json.dumps(vars(a) | {"final_ema_loss": ema, "steps": steps}, indent=2))
        log(f"saved adapter to {out}")
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
