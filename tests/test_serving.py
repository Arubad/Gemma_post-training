import json

import yaml

from gemma_swe.constants import REPO_ROOT
from gemma_swe.serving import eval_config, set_seed, swegemma_cmd, vllm_command, vllm_max_lora_rank
from gemma_swe.submission_builder import build_dir, load_profile


def test_vllm_command_matches_scorer_flags(tmp_path):
    a = tmp_path / "lora"
    a.mkdir()
    (a / "adapter_config.json").write_text(json.dumps({"r": 24}))
    cmd = vllm_command("/m", "gemma-4-12b-it-qat-w4a16-ct", {"main_lora": str(a)}, port=8001)
    s = " ".join(cmd)
    for flag in ("--tensor-parallel-size 4", "--max-model-len 32768", "--tool-call-parser gemma4",
                 "--reasoning-parser gemma4", "--enable-auto-tool-choice", "--enable-lora", "--max-loras 1",
                 "--max-lora-rank 32", f"main_lora={a}", "--hf-overrides"):
        assert flag in s, flag
    assert vllm_max_lora_rank([4, 16]) == 16 and vllm_max_lora_rank([33]) == 64


def test_swegemma_cmd_and_seed(tmp_path):
    sub = build_dir(load_profile(REPO_ROOT / "profiles" / "e4b_prompt.yaml"), tmp_path / "sub")
    ev = eval_config(sub)
    cmd = swegemma_cmd("t.jsonl", "snaps", str(sub), "res", 4, ev, task_ids=["a", "b"])
    s = " ".join(cmd)
    assert "--sandbox subprocess" in s and "--max-tool-calls 40" in s and "--max-time-minutes 5" in s and "--task-ids a b" in s
    set_seed(sub, 7, temperature=0.7)
    d = yaml.safe_load((sub / "configs" / "sampling.yaml").read_text())
    assert d["seed"] == 7 and d["temperature"] == 0.7 and d["thinking_config"]["thinking_budget"] == 0
