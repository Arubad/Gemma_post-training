# %% [markdown]
# # NB04 · Scorer-faithful local evaluation on the held-out repo (L4x4, internet OFF)
# **Inputs:** competition data + wheelhouse, Kaggle Model, `gemma-swe-code`, `gemma-swe-data`, `gemma-swe-adapters`
# **GPU-h:** ~1 per seed for the 48 held-out tasks at concurrency 4 (x2 quota)
#
# One public-LB task = 0.017 and identical zips have scored 0.10 vs 0.15, so decisions are made here:
# >= 3 seeds, mean +- std, per hop bucket. Compare every arm against the prompt-only control.

# %%
import os, subprocess, sys
CODE = "/kaggle/input/gemma-swe-code"
sys.path.insert(0, CODE)
from gemma_swe.kaggle_env import find_comp_dir, find_model_dir, install_harness
COMP, DATA = find_comp_dir(), "/kaggle/input/gemma-swe-data"
CKPT = find_model_dir("gemma-4-e4b-it-qat-w4a16-ct")
ADAPTER = os.environ.get("ADAPTER", "/kaggle/input/gemma-swe-adapters/sft_v1")
TASKS = f"{DATA}/tasks_heldout.jsonl"
SEEDS = "0 1 2"

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

install_harness()

# %%
arms = {
    "e4b_prompt": f"--profile {CODE}/profiles/e4b_prompt.yaml",
    "e4b_lora": f"--profile {CODE}/profiles/e4b_lora.yaml --adapter main_lora={ADAPTER}",
}
for name, args in arms.items():
    sh(f"python {CODE}/scripts/build_submission.py {args} --dir /kaggle/working/sub_{name}")
    sh(f"python {CODE}/scripts/local_eval.py --submission-dir /kaggle/working/sub_{name} --model-dir {CKPT} "
       f"--tasks {TASKS} --snapshots-dir {COMP}/snapshots --out /kaggle/working/eval/{name} --seeds {SEEDS} --concurrency 4")

# %%
runs = " ".join(f"--run {n}=/kaggle/working/eval/{n}" for n in arms)
sh(f"python {CODE}/scripts/eval_report.py {runs} --tasks {TASKS} --hop-map {DATA}/hop_map.json | tee /kaggle/working/eval/report.json")
# If adapter requests show the base model name in `adapter_used`, the CLI is not routing adapters:
# re-run local_eval.py with --route-base-to main_lora.
