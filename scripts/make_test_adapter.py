#!/usr/bin/env python
"""Create a zero-init (B=0) or random LoRA adapter directly from a checkpoint's weight shapes.

Used for the serving smoke tests before any training (report: "first GPU session"):
  * zero adapter  -> vLLM outputs with the adapter must EQUAL the base (adapter path works)
  * random adapter -> outputs must DIFFER (adapter is actually applied, not silently ignored)

    python scripts/make_test_adapter.py --ckpt $BF16_DIR --out /kaggle/tmp/zero_lora --mode zero
    python scripts/make_test_adapter.py --ckpt $BF16_DIR --out /kaggle/tmp/rand_lora --mode random

Works from the dequantized bf16 dir (plain .weight shapes) or the w4a16 dir (weight_shape is read).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.adapter_utils import TARGET_RE, kv_shared_layers, read_safetensors_header, text_config  # noqa: E402


def module_shapes(ckpt: Path) -> dict[str, tuple[int, int]]:
    import torch  # noqa: F401
    from safetensors import safe_open

    shapes = {}
    for f in sorted(ckpt.glob("*.safetensors")):
        hdr = read_safetensors_header(f)
        packed = {k for k in hdr if k.endswith(".weight_shape")}
        with safe_open(str(f), framework="pt") as sf:
            for k, meta in hdr.items():
                if k.endswith(".weight") and TARGET_RE.search(k[:-7]):
                    shapes[k[:-7]] = tuple(meta["shape"])
                elif k in packed and TARGET_RE.search(k[: -len(".weight_shape")]):
                    shapes[k[: -len(".weight_shape")]] = tuple(int(x) for x in sf.get_tensor(k).tolist())
    return shapes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["zero", "random"], default="zero")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=0.02)
    a = ap.parse_args(argv)
    import torch
    from safetensors.torch import save_file

    ckpt = Path(a.ckpt)
    shared = kv_shared_layers(text_config(ckpt))
    tensors = {}
    g = torch.Generator().manual_seed(0)
    for mod, (out_f, in_f) in sorted(module_shapes(ckpt).items()):
        if any(x in mod for x in ("vision", "audio")):
            continue
        t = TARGET_RE.search(mod)
        if int(t.group("layer")) in shared and t.group("proj") in ("k_proj", "v_proj"):
            continue
        A = torch.randn(a.rank, in_f, generator=g) * (1 / in_f**0.5)
        B = torch.zeros(out_f, a.rank) if a.mode == "zero" else torch.randn(out_f, a.rank, generator=g) * a.scale
        tensors[f"base_model.model.{mod}.lora_A.weight"] = A.to(torch.bfloat16).contiguous()
        tensors[f"base_model.model.{mod}.lora_B.weight"] = B.to(torch.bfloat16).contiguous()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    projs = sorted({re.search(r"\.(\w+_proj)\.lora_A", k).group(1) for k in tensors if ".lora_A." in k})
    (out / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": a.rank, "lora_alpha": a.rank, "lora_dropout": 0.0,
        "target_modules": projs, "bias": "none", "base_model_name_or_path": str(ckpt), "inference_mode": True,
        "fan_in_fan_out": False, "modules_to_save": None, "use_dora": False, "use_rslora": False,
    }, indent=2))
    print(f"wrote {len(tensors) // 2} LoRA modules ({a.mode}) to {out}")


if __name__ == "__main__":
    main()
