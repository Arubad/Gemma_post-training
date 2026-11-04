"""Inspect / validate / repair PEFT LoRA adapters against the served w4a16 checkpoint.

Pure Python safetensors header parsing (no torch needed for validation); key renaming
needs `safetensors` + `torch`.

Failure modes this guards against (all observed in the competition, see report section 6):
  * adapter keys that do not match the served module names -> vLLM silently ignores the LoRA
  * LoRA on modules vLLM drops (E4B KV-shared layers' k/v_proj, 31B global layers' v_proj)
  * vision/audio tower or embedding / lm_head tensors in the adapter
  * rank > 128, non-safetensors weights, wrong file names
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path

LORA_KEY_RE = re.compile(r"^(?P<module>.+)\.lora_(?P<ab>A|B)(?:\.(?P<adapter>[^.]+))?\.weight$")
TARGET_RE = re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\.(?P<block>self_attn|mlp)\.(?P<proj>[qkvo]_proj|gate_proj|up_proj|down_proj)$")
FORBIDDEN_PARTS = ("vision", "audio", "embed", "lm_head", "multi_modal", "per_layer")


def read_safetensors_header(path: str | Path) -> dict:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def checkpoint_weight_names(ckpt_dir: str | Path) -> set[str]:
    ckpt_dir = Path(ckpt_dir)
    idx = ckpt_dir / "model.safetensors.index.json"
    if idx.exists():
        return set(json.loads(idx.read_text())["weight_map"])
    names: set[str] = set()
    for f in sorted(ckpt_dir.glob("*.safetensors")):
        names |= set(read_safetensors_header(f))
    return names


def checkpoint_linear_modules(ckpt_dir: str | Path) -> set[str]:
    """Module paths that have a (packed or plain) linear weight in the checkpoint."""
    mods = set()
    for n in checkpoint_weight_names(ckpt_dir):
        for suf in (".weight_packed", ".weight"):
            if n.endswith(suf):
                mods.add(n[: -len(suf)])
    return mods


def text_config(ckpt_dir: str | Path) -> dict:
    cfg = json.loads((Path(ckpt_dir) / "config.json").read_text())
    return cfg.get("text_config", cfg)


def kv_shared_layers(tcfg: dict) -> set[int]:
    n = int(tcfg.get("num_hidden_layers", 0))
    k = int(tcfg.get("num_kv_shared_layers", 0) or 0)
    return set(range(n - k, n)) if k else set()


def adapter_module(key: str) -> tuple[str, str] | None:
    m = LORA_KEY_RE.match(key)
    if not m:
        return None
    mod = m.group("module")
    if mod.startswith("base_model.model."):
        mod = mod[len("base_model.model.") :]
    return mod, m.group("ab")


def validate_adapter(adapter_dir: str | Path, base_ckpt_dir: str | Path | None = None) -> dict:
    adapter_dir = Path(adapter_dir)
    rep: dict = {"ok": True, "errors": [], "warnings": [], "n_tensors": 0, "rank": None, "modules": 0}

    def err(msg):
        rep["ok"] = False
        rep["errors"].append(msg)

    cfg_p = adapter_dir / "adapter_config.json"
    w_p = adapter_dir / "adapter_model.safetensors"
    if not cfg_p.exists():
        err("missing adapter_config.json")
    if not w_p.exists():
        err("missing adapter_model.safetensors")
    for f in adapter_dir.iterdir() if adapter_dir.exists() else []:
        if f.suffix not in (".json", ".safetensors"):
            if f.suffix in (".md", ".txt"):
                rep["warnings"].append(f"extra file {f.name} (not copied into the submission)")
            else:
                err(f"disallowed file in adapter dir: {f.name}")
    if not rep["ok"]:
        return rep
    cfg = json.loads(cfg_p.read_text())
    r = int(cfg.get("r", 0))
    rep["rank"] = r
    alpha = cfg.get("lora_alpha")
    if r < 1 or r > 128:
        err(f"rank {r} outside 1..128")
    if alpha is not None and r and alpha / r >= 2:
        rep["warnings"].append(f"lora_alpha/r = {alpha / r:.1f}; alpha/r=2 was reported to induce loops")
    if cfg.get("use_dora"):
        err("DoRA adapters are not supported by vLLM")
    if cfg.get("modules_to_save"):
        err("modules_to_save is set (full-weight modules are not served)")

    hdr = read_safetensors_header(w_p)
    rep["n_tensors"] = len(hdr)
    mods: dict[str, set[str]] = {}
    for k, meta in hdr.items():
        am = adapter_module(k)
        if am is None:
            err(f"non-LoRA tensor in adapter: {k}")
            continue
        mod, ab = am
        mods.setdefault(mod, set()).add(ab)
        if any(p in mod for p in FORBIDDEN_PARTS):
            err(f"LoRA on forbidden module: {mod}")
        if not TARGET_RE.search(mod):
            rep["warnings"].append(f"unexpected target module: {mod}")
        shape = meta.get("shape", [])
        if ab == "A" and shape and shape[0] != r:
            err(f"{k}: lora_A rows {shape[0]} != r {r}")
        if meta.get("dtype") not in ("BF16", "F16", "F32"):
            rep["warnings"].append(f"{k}: dtype {meta.get('dtype')}")
    for mod, abs_ in mods.items():
        if abs_ != {"A", "B"}:
            err(f"{mod}: missing lora_{'B' if 'A' in abs_ else 'A'}")
    rep["modules"] = len(mods)

    if base_ckpt_dir:
        base_mods = checkpoint_linear_modules(base_ckpt_dir)
        tcfg = text_config(base_ckpt_dir)
        shared = kv_shared_layers(tcfg)
        missing = [m for m in mods if m not in base_mods]
        if missing:
            err(f"{len(missing)} adapter modules do not exist in the base checkpoint, e.g. {missing[:3]} "
                f"(base example: {sorted(base_mods)[:2]}) -> run fix_adapter_keys")
        for m in mods:
            t = TARGET_RE.search(m)
            if t and int(t.group("layer")) in shared and t.group("proj") in ("k_proj", "v_proj"):
                err(f"LoRA on KV-shared layer k/v projection (dropped by vLLM): {m}")
    return rep


# ------------------------------------------------------------------------------- repairs
def _candidate_rewrites(mod: str) -> list[str]:
    out = [mod]
    swaps = [
        ("model.language_model.model.", "model.language_model."),
        ("model.language_model.", "model.language_model."),
        ("language_model.model.", "model.language_model."),
        ("model.model.", "model.language_model."),
        ("model.", "model.language_model."),
        ("model.language_model.", "model."),
    ]
    for a, b in swaps:
        if mod.startswith(a):
            out.append(b + mod[len(a) :])
    return out


def plan_key_fixes(adapter_dir: str | Path, base_ckpt_dir: str | Path) -> dict[str, str]:
    """old_module -> new_module mapping so that every adapter module exists in the base."""
    base_mods = checkpoint_linear_modules(base_ckpt_dir)
    hdr = read_safetensors_header(Path(adapter_dir) / "adapter_model.safetensors")
    plan: dict[str, str] = {}
    for k in hdr:
        am = adapter_module(k)
        if not am:
            continue
        mod = am[0]
        if mod in plan:
            continue
        for cand in _candidate_rewrites(mod):
            if cand in base_mods:
                plan[mod] = cand
                break
        else:
            plan[mod] = None  # type: ignore[assignment]
    return plan


def fix_adapter(adapter_dir: str | Path, base_ckpt_dir: str | Path, out_dir: str | Path,
                drop_kv_shared: bool = True) -> dict:
    """Rename keys to the base checkpoint's module names, drop unservable tensors, normalise
    adapter_config.json. Writes a clean PEFT dir to out_dir."""
    import torch  # noqa: F401
    from safetensors.torch import load_file, save_file

    adapter_dir, out_dir = Path(adapter_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_key_fixes(adapter_dir, base_ckpt_dir)
    shared = kv_shared_layers(text_config(base_ckpt_dir)) if drop_kv_shared else set()
    tensors = load_file(str(adapter_dir / "adapter_model.safetensors"))
    new, dropped = {}, []
    for k, v in tensors.items():
        am = adapter_module(k)
        if not am:
            dropped.append(k)
            continue
        mod, ab = am
        tgt = plan.get(mod)
        if tgt is None:
            dropped.append(k)
            continue
        t = TARGET_RE.search(tgt)
        if t and int(t.group("layer")) in shared and t.group("proj") in ("k_proj", "v_proj"):
            dropped.append(k)
            continue
        if any(p in tgt for p in FORBIDDEN_PARTS):
            dropped.append(k)
            continue
        new[f"base_model.model.{tgt}.lora_{ab}.weight"] = v.contiguous().to(torch.bfloat16)
    save_file(new, str(out_dir / "adapter_model.safetensors"), metadata={"format": "pt"})

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    procs = sorted({TARGET_RE.search(k[: -len('.lora_A.weight')]).group("proj")
                    for k in new if k.endswith(".lora_A.weight") and TARGET_RE.search(k[: -len('.lora_A.weight')])})
    cfg["target_modules"] = procs
    for key in ("layers_to_transform", "layers_pattern", "rank_pattern", "alpha_pattern", "exclude_modules"):
        cfg.pop(key, None)
    cfg["modules_to_save"] = None
    cfg["use_dora"] = False
    cfg["bias"] = "none"
    cfg["inference_mode"] = True
    (out_dir / "adapter_config.json").write_text(json.dumps(cfg, indent=2))
    return {"kept": len(new), "dropped": len(dropped), "dropped_examples": dropped[:5],
            "renamed": sum(1 for a, b in plan.items() if b and a != b)}


def adapter_size_bytes(adapter_dir: str | Path) -> int:
    return sum(f.stat().st_size for f in Path(adapter_dir).iterdir() if f.is_file())
