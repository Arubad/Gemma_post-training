# %% [markdown]
# # NB06 · Package the two final submissions (CPU, internet OFF)
# **Inputs:** `gemma-swe-code`, `gemma-swe-adapters`, Kaggle Models (for adapter key validation) · **GPU-h:** 0
#
# Final 1 (small-model bonus + main board): E4B + best adapter by NB04 held-out mean over >= 3 seeds.
# Final 2 (main board): 31B prompt-only (or `31b_lora` if a 31B adapter beat it locally).
# Submit each `submission_*.zip` on the competition page (1/day). Do not chase +-1 task public-LB swings.

# %%
import os, subprocess, sys
CODE = "/kaggle/input/gemma-swe-code"
sys.path.insert(0, CODE)
from gemma_swe.kaggle_env import find_model_dir

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

BEST_E4B = os.environ.get("BEST_E4B", "/kaggle/input/gemma-swe-adapters/rft1")
E4B = find_model_dir("gemma-4-e4b-it-qat-w4a16-ct")

# %%
sh(f"python {CODE}/scripts/build_submission.py --profile {CODE}/profiles/e4b_lora.yaml --adapter main_lora={BEST_E4B} "
   f"--base-ckpt {E4B} --dir /kaggle/working/final_e4b --zip /kaggle/working/submission_e4b_lora.zip")
sh(f"python {CODE}/scripts/build_submission.py --profile {CODE}/profiles/31b_prompt.yaml "
   f"--dir /kaggle/working/final_31b --zip /kaggle/working/submission_31b_prompt.zip")
sh("ls -la /kaggle/working/*.zip")
