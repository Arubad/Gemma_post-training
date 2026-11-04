import json
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from gemma_swe.adapter_utils import fix_adapter, validate_adapter
from gemma_swe.constants import REPO_ROOT
from gemma_swe.submission_builder import SubmissionError, build_dir, load_profile, load_yaml, validate_dir, zip_dir

N_LAYERS, SHARED, H = 4, 2, 8


def fake_base(tmp: Path) -> Path:
    """Header-compatible fake w4a16 checkpoint: packed linears for 4 layers, last 2 KV-shared."""
    d = tmp / "base"
    d.mkdir()
    t = {}
    for i in range(N_LAYERS):
        for blk, projs in (("self_attn", "qkvo"), ("mlp", None)):
            names = [f"{p}_proj" for p in projs] if projs else ["gate_proj", "up_proj", "down_proj"]
            for n in names:
                k = f"model.language_model.layers.{i}.{blk}.{n}"
                t[k + ".weight_packed"] = torch.zeros(H, 1, dtype=torch.int32)
                t[k + ".weight_scale"] = torch.ones(H, 1, dtype=torch.bfloat16)
                t[k + ".weight_shape"] = torch.tensor([H, H])
    t["model.language_model.embed_tokens.weight"] = torch.zeros(10, H, dtype=torch.bfloat16)
    save_file(t, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": N_LAYERS, "num_kv_shared_layers": SHARED}}))
    return d


def fake_adapter(tmp: Path, prefix="base_model.model.model.language_model.", r=4, kv_shared=False, name="lora") -> Path:
    d = tmp / name
    d.mkdir()
    t = {}
    for i in range(N_LAYERS):
        for p in ("q_proj", "k_proj", "o_proj", "gate_proj", "down_proj"):
            if p == "k_proj" and i >= N_LAYERS - SHARED and not kv_shared:
                continue
            blk = "self_attn" if p in ("q_proj", "k_proj", "o_proj") else "mlp"
            m = f"{prefix}layers.{i}.{blk}.{p}"
            t[f"{m}.lora_A.weight"] = torch.randn(r, H).to(torch.bfloat16)
            t[f"{m}.lora_B.weight"] = torch.zeros(H, r, dtype=torch.bfloat16)
    save_file(t, str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": r, "peft_type": "LORA", "target_modules": ["q_proj"]}))
    return d


def test_adapter_validate_and_fix(tmp_path):
    base = fake_base(tmp_path)
    good = fake_adapter(tmp_path)
    assert validate_adapter(good, base)["ok"]
    bad = fake_adapter(tmp_path, prefix="base_model.model.model.", kv_shared=True, name="bad")
    rep = validate_adapter(bad, base)
    assert not rep["ok"] and any("do not exist" in e for e in rep["errors"])
    out = tmp_path / "fixed"
    info = fix_adapter(bad, base, out)
    assert info["renamed"] > 0 and info["dropped"] == 2 * SHARED  # KV-shared k_proj A+B dropped
    rep2 = validate_adapter(out, base)
    assert rep2["ok"], rep2["errors"]
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert sorted(cfg["target_modules"]) == ["down_proj", "gate_proj", "k_proj", "o_proj", "q_proj"]


@pytest.mark.parametrize("profile", sorted(p.name for p in (REPO_ROOT / "profiles").glob("*.yaml")))
def test_build_all_profiles(tmp_path, profile, monkeypatch):
    base = fake_base(tmp_path)
    monkeypatch.setenv("ADAPTER_DIR", str(fake_adapter(tmp_path)))
    prof = load_profile(REPO_ROOT / "profiles" / profile)
    sub = build_dir(prof, tmp_path / "sub")
    rep = validate_dir(sub, base)
    assert rep["ok"], rep["errors"]
    assert rep["models"] == [prof["model_name"]]
    agent = load_yaml(sub / "agent.yaml", sub)
    tc = agent["generate_content_config"]["thinking_config"]
    assert "submit_patch" in agent["tools"]
    assert tc["include_thoughts"] == (tc["thinking_budget"] > 0)  # README: vLLM+LoRA needs include_thoughts + budget
    assert "{" not in agent["instruction"]
    z = zip_dir(sub, tmp_path / "submission.zip")
    names = zipfile.ZipFile(z).namelist()
    assert "agent.yaml" in names and "eval_config.yaml" in names and all(not n.startswith("/") for n in names)
    if prof.get("variant") == "localizer":
        loc = load_yaml(sub / "sub_agents" / "localizer.yaml", sub)
        assert loc["model"] == prof["model_name"] and "edit_file" not in loc["tools"]


def test_validation_catches_mistakes(tmp_path):
    prof = load_profile(REPO_ROOT / "profiles" / "e4b_prompt.yaml")
    sub = build_dir(prof, tmp_path / "s")
    (sub / "prompts" / "system.md").write_text("Fix {repo_name} now")
    rep = validate_dir(sub)
    assert not rep["ok"] and any("placeholder" in e for e in rep["errors"])
    sub = build_dir(prof, tmp_path / "s2")
    (sub / "agent.yaml").write_text((sub / "agent.yaml").read_text().replace("  - submit_patch", "  - submit_patch\n  - web_search"))
    assert any("web_search" in e for e in validate_dir(sub)["errors"])
    sub = build_dir(prof, tmp_path / "s3")
    (sub / "agent.yaml").write_text((sub / "agent.yaml").read_text().replace("!include prompts/system.md", "!include ../x.md"))
    assert not validate_dir(sub)["ok"]
    sub = build_dir(prof, tmp_path / "s4")
    (sub / "weights.bin").write_bytes(b"x")
    assert any("extension" in e for e in validate_dir(sub)["errors"])
    with pytest.raises(SubmissionError):
        build_dir(load_profile(REPO_ROOT / "profiles" / "e4b_lora.yaml"), tmp_path / "s5", {"main_lora": str(tmp_path / "nope")})
