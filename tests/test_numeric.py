import json

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from gemma_swe.dequant import _unpack_local, dequant_weight, dequantize_checkpoint, pack_int4_reference
from gemma_swe.losses import chunked_lm_loss, shift_labels


def test_int4_pack_unpack_roundtrip():
    q = torch.randint(-8, 8, (6, 64), dtype=torch.int8)
    packed = pack_int4_reference(q)
    assert packed.shape == (6, 8) and packed.dtype == torch.int32
    assert torch.equal(_unpack_local(packed, 4, (6, 64)), q)


def test_dequant_checkpoint(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    q = torch.randint(-8, 8, (4, 64), dtype=torch.int8)
    scale = (torch.rand(4, 2) + 0.5).to(torch.bfloat16)  # group size 32
    save_file({
        "model.layers.0.mlp.up_proj.weight_packed": pack_int4_reference(q),
        "model.layers.0.mlp.up_proj.weight_scale": scale,
        "model.layers.0.mlp.up_proj.weight_shape": torch.tensor([4, 64]),
        "model.norm.weight": torch.ones(64, dtype=torch.bfloat16),
    }, str(src / "model.safetensors"))
    (src / "config.json").write_text(json.dumps({"quantization_config": {"config_groups": {"g": {"weights": {"num_bits": 4}}}}}))
    (src / "tokenizer.json").write_text("{}")
    info = dequantize_checkpoint(src, dst)
    assert info["dequantized_linears"] == 1
    out = {}
    for f in dst.glob("*.safetensors"):
        out.update(load_file(str(f)))
    w = out["model.layers.0.mlp.up_proj.weight"]
    ref = (q.float() * scale.float().repeat_interleave(32, dim=1)).to(torch.bfloat16)
    assert torch.equal(w, ref) and torch.equal(w, dequant_weight(pack_int4_reference(q), scale, (4, 64)))
    assert set(out) == {"model.layers.0.mlp.up_proj.weight", "model.norm.weight"}
    assert "quantization_config" not in json.loads((dst / "config.json").read_text())
    assert (dst / "tokenizer.json").exists()


def test_chunked_loss_matches_full():
    torch.manual_seed(0)
    B, T, Hd, V = 1, 37, 16, 50
    h = torch.randn(B, T, Hd, requires_grad=True)
    w = torch.randn(V, Hd, requires_grad=True)
    ids = torch.randint(0, V, (B, T))
    mask = (torch.rand(B, T) > 0.4).long()
    lab = shift_labels(ids, mask)
    s, n = chunked_lm_loss(h, w, lab, softcap=30.0, chunk_size=7)
    logits = torch.tanh((h @ w.t()) / 30.0) * 30.0
    ref = F.cross_entropy(logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100, reduction="sum")
    assert n == int((lab != -100).sum())
    assert torch.allclose(s, ref, atol=1e-4)
    gh, gw = torch.autograd.grad(s, (h, w))
    rh, rw = torch.autograd.grad(ref, (h, w))
    assert torch.allclose(gh, rh, atol=1e-4) and torch.allclose(gw, rw, atol=1e-4)
    assert lab[0, -1] == -100 and (lab[0, :-1][mask[0, 1:] == 1] == ids[0, 1:][mask[0, 1:] == 1]).all()
