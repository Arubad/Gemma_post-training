"""Dequantize a compressed-tensors `pack-quantized` W4A16 checkpoint to bf16 safetensors.

Training the LoRA on these weights means the frozen base during training is numerically the
same int4-grid weights Marlin reconstructs at serve time (train == serve up to accumulation
order), which is the recommended recipe (report section "Train on dequantized w4a16 weights").

Format (Gemma 4 *-qat-w4a16-ct): every quantized Linear `X` stores
    X.weight_packed  int32 [out, in/8]   8 int4 values per int32, little-endian nibbles,
                                          stored with +8 offset (symmetric)
    X.weight_scale   bf16  [out, in/group_size]
    X.weight_shape   int   [2]            (original shape)
    X.weight_zero_point (optional, asymmetric only)
dequant: W = (q - zp) * scale, broadcast per group along the input dim.

Uses compressed_tensors' own unpack_from_int32 when importable (exactness by construction);
otherwise a local re-implementation that `verify_against_library()` cross-checks.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


def _unpack_local(packed: torch.Tensor, num_bits: int, shape: tuple[int, int]) -> torch.Tensor:
    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    rows, cols = shape
    shifts = torch.arange(pack_factor, dtype=torch.int32, device=packed.device) * num_bits
    un = (packed.to(torch.int32).unsqueeze(-1) >> shifts) & mask  # [rows, packed_cols, pf]
    un = un.reshape(packed.shape[0], -1)[:, :cols]
    offset = 1 << (num_bits - 1)
    return (un - offset).to(torch.int8)


def unpack_int4(packed: torch.Tensor, shape: tuple[int, int], num_bits: int = 4) -> torch.Tensor:
    try:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32

        return unpack_from_int32(packed, num_bits, torch.Size(shape))
    except Exception:
        return _unpack_local(packed, num_bits, shape)


def dequant_weight(packed: torch.Tensor, scale: torch.Tensor, shape: tuple[int, int],
                   zero_point: torch.Tensor | None = None, num_bits: int = 4,
                   out_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    q = unpack_int4(packed, shape, num_bits).to(torch.float32)
    rows, cols = shape
    group = cols // scale.shape[1]
    s = scale.to(torch.float32).repeat_interleave(group, dim=1)[:, :cols]
    if zero_point is not None and zero_point.numel():
        zp = zero_point.to(torch.float32)
        if zp.shape != scale.shape:  # packed zero points
            zp = unpack_int4(zero_point, tuple(scale.shape), num_bits).to(torch.float32)
        q = q - zp.repeat_interleave(group, dim=1)[:, :cols]
    return (q * s).to(out_dtype)


def _quant_cfg(cfg: dict) -> dict:
    qc = cfg.get("quantization_config") or cfg.get("compression_config") or {}
    return qc


def dequantize_checkpoint(src: str | Path, dst: str | Path, num_bits: int | None = None,
                          max_shard_bytes: int = 4 * 1024**3) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    qc = _quant_cfg(cfg)
    if num_bits is None:
        try:
            num_bits = int(next(iter(qc["config_groups"].values()))["weights"]["num_bits"])
        except Exception:
            num_bits = 4

    shards = sorted(src.glob("*.safetensors"))
    weight_map: dict[str, str] = {}
    out_buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    shard_idx = 0
    n_deq = 0

    def flush():
        nonlocal out_buf, buf_bytes, shard_idx
        if not out_buf:
            return
        shard_idx += 1
        name = f"model-{shard_idx:05d}.safetensors"
        save_file(out_buf, str(dst / name), metadata={"format": "pt"})
        for k in out_buf:
            weight_map[k] = name
        out_buf, buf_bytes = {}, 0

    for sh in shards:
        with safe_open(str(sh), framework="pt") as f:
            keys = list(f.keys())
            kset = set(keys)
            for k in keys:
                if k.endswith((".weight_scale", ".weight_shape", ".weight_zero_point", ".weight_g_idx")):
                    base = k.rsplit(".", 1)[0]
                    if f"{base}.weight_packed" in kset:
                        continue
                if k.endswith(".weight_packed"):
                    base = k[: -len(".weight_packed")]
                    packed = f.get_tensor(k)
                    scale = f.get_tensor(f"{base}.weight_scale")
                    if f"{base}.weight_shape" in kset:
                        shape = tuple(int(x) for x in f.get_tensor(f"{base}.weight_shape").tolist())
                    else:
                        shape = (packed.shape[0], packed.shape[1] * (32 // num_bits))
                    zp = f.get_tensor(f"{base}.weight_zero_point") if f"{base}.weight_zero_point" in kset else None
                    t = dequant_weight(packed, scale, shape, zp, num_bits)
                    name = f"{base}.weight"
                    n_deq += 1
                else:
                    t = f.get_tensor(k)
                    name = k
                out_buf[name] = t.contiguous()
                buf_bytes += t.numel() * t.element_size()
                if buf_bytes >= max_shard_bytes:
                    flush()
    flush()

    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum((dst / v).stat().st_size for v in set(weight_map.values()))},
                    "weight_map": weight_map}, indent=2)
    )
    cfg.pop("quantization_config", None)
    cfg.pop("compression_config", None)
    cfg["torch_dtype"] = "bfloat16"
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))
    for f in src.iterdir():
        if f.suffix in (".safetensors",) or f.name in ("config.json", "model.safetensors.index.json"):
            continue
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    return {"dequantized_linears": n_deq, "shards": shard_idx, "num_bits": num_bits}


def verify_against_library(src: str | Path, n_layers: int = 4) -> list[dict]:
    """Compare local unpack vs compressed_tensors' for a few packed weights (should be exact)."""
    from safetensors import safe_open

    out = []
    try:
        from compressed_tensors.compressors.quantized_compressors.pack_quantized import unpack_from_int32
    except Exception as e:  # library absent: nothing to compare
        return [{"skipped": f"compressed_tensors not importable: {e}"}]
    for sh in sorted(Path(src).glob("*.safetensors")):
        with safe_open(str(sh), framework="pt") as f:
            for k in f.keys():
                if not k.endswith(".weight_packed"):
                    continue
                base = k[: -len(".weight_packed")]
                packed = f.get_tensor(k)
                shape = tuple(int(x) for x in f.get_tensor(f"{base}.weight_shape").tolist())
                a = _unpack_local(packed, 4, shape)
                b = unpack_from_int32(packed, 4, torch.Size(shape))
                out.append({"tensor": base, "equal": bool(torch.equal(a, b.to(a.dtype)))})
                if len(out) >= n_layers:
                    return out
    return out


def pack_int4_reference(q: torch.Tensor) -> torch.Tensor:
    """Reference packer (same layout as compressed_tensors.pack_to_int32), used by the tests."""
    rows, cols = q.shape
    pf = 8
    v = (q.to(torch.int32) + 8)
    pad = (-cols) % pf
    if pad:
        v = torch.nn.functional.pad(v, (0, pad))
    v = v.view(rows, -1, pf)
    shifts = torch.arange(pf, dtype=torch.int32) * 4
    return (v << shifts).sum(dim=2, dtype=torch.int32)
