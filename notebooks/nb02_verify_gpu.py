# %% [markdown]
# # NB02 · First GPU session = verification, not training (L4x4, internet OFF)
# **Accelerator:** GPU L4 x4 · **Inputs:** competition data + wheelhouse, Kaggle Model `google/gemma-4`
# (e4b-it-qat-w4a16-ct), `gemma-swe-code`, `gemma-swe-data`, `gemma-swe-wheels` · **GPU-h:** ~1.5 (x2 quota)
#
# Every later number rests on these checks (report, final section). Do not train until all pass:
# 1. dequantized bf16 == served int4 grid (library cross-check)
# 2. zero-init adapter -> outputs identical to base; random adapter -> different (LoRA really applied in vLLM)
# 3. our training render == vLLM's own prompt (/tokenize) for real records
# 4. gemma4 tool parser round-trips multi-line edit_file / heredoc arguments
# 5. untuned E4B prompt-only baseline on the smoke tasks (loops / submits / malformed calls)

# %%
import json, os, subprocess, sys, time
CODE = "/kaggle/input/gemma-swe-code"
sys.path.insert(0, CODE)
from gemma_swe.kaggle_env import find_comp_dir, find_model_dir, find_wheel_dir, gpu_summary, install_harness, pip_offline
COMP, DATA = find_comp_dir(), "/kaggle/input/gemma-swe-data"
CKPT = find_model_dir("gemma-4-e4b-it-qat-w4a16-ct")
BF16, TMP = "/kaggle/tmp/e4b-bf16", "/kaggle/tmp"
NAME = "gemma-4-e4b-it-qat-w4a16-ct"
print(gpu_summary(), CKPT)

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    return subprocess.run(cmd, shell=True, check=check)

install_harness()
pip_offline(["peft", "compressed-tensors"], [find_wheel_dir("peft-*.whl")], no_deps=True)

# %% 1. dequantize (+ cross-check unpacker against compressed_tensors)
sh(f"python {CODE}/scripts/dequantize_checkpoint.py --src {CKPT} --dst {BF16} --verify")

# %% 2. adapter serving smoke test
sh(f"python {CODE}/scripts/make_test_adapter.py --ckpt {BF16} --out {TMP}/zero_lora --mode zero")
sh(f"python {CODE}/scripts/make_test_adapter.py --ckpt {BF16} --out {TMP}/rand_lora --mode random")
sh(f"python {CODE}/scripts/check_adapter.py --adapter {TMP}/zero_lora --base-ckpt {CKPT}")
srv = subprocess.Popen(f"python {CODE}/scripts/serve_vllm.py --model-dir {CKPT} --served-name {NAME} "
                       f"--adapter zero_lora={TMP}/zero_lora --adapter rand_lora={TMP}/rand_lora --port 8001 --log {TMP}/vllm.log",
                       shell=True)
srv.wait()  # returns once READY (vLLM keeps running in its own process group)
EP = "http://127.0.0.1:8001"
sh(f"python {CODE}/scripts/verify_serving.py adapter --endpoint {EP} --model {NAME} --adapter zero_lora --expect same")
sh(f"python {CODE}/scripts/verify_serving.py adapter --endpoint {EP} --model {NAME} --adapter rand_lora --expect different")
sh(f"grep -i -E 'lora|kv cache|maximum concurrency' {TMP}/vllm.log | tail -20", check=False)

# %% 3. render fidelity vs the live server, 4. tool parser round-trip
sh(f"python {CODE}/scripts/verify_serving.py render --endpoint {EP} --tokenizer {CKPT} --model {NAME} "
   f"--record-file {DATA}/graph_synth.jsonl --calibration {DATA}/calibration.json --profile {CODE}/profiles/e4b_lora.yaml --n 5")
sh(f"python {CODE}/scripts/verify_serving.py render --endpoint {EP} --tokenizer {CKPT} --model {NAME} "
   f"--record-file {DATA}/remapped.jsonl --calibration {DATA}/calibration.json --profile {CODE}/profiles/e4b_lora.yaml --n 5")
sh(f"python {CODE}/scripts/verify_serving.py parser --tokenizer {CKPT} --data {DATA}/remapped.jsonl --n 300", check=False)
sh("pkill -f vllm.entrypoints.openai.api_server", check=False); time.sleep(20)

# %% 5. untuned E4B prompt-only baseline on smoke tasks (also benchmarks wall-clock per task)
sh(f"python {CODE}/scripts/build_submission.py --profile {CODE}/profiles/e4b_prompt.yaml --dir /kaggle/working/sub_e4b_prompt")
sh(f"python {CODE}/scripts/local_eval.py --submission-dir /kaggle/working/sub_e4b_prompt --model-dir {CKPT} "
   f"--tasks {DATA}/tasks_smoke.jsonl --snapshots-dir {COMP}/snapshots --out /kaggle/working/eval/smoke_e4b_prompt --seeds 0 --concurrency 2")
sh(f"python {CODE}/scripts/eval_report.py --run e4b_prompt=/kaggle/working/eval/smoke_e4b_prompt "
   f"--tasks {DATA}/tasks_smoke.jsonl --hop-map {DATA}/hop_map.json")
