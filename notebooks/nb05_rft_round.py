# %% [markdown]
# # NB05 · One round of rejection-sampling fine-tuning in the exact harness (L4x4, internet OFF)
# **GPU-h:** ~6-10 rollouts (8 seeds x 80 train tasks at temperature 0.7) + ~3 training (x2 quota).
# Split rollouts across accounts by seed (`SEEDS` env var) and merge the harvested JSONL files.
#
# Rollouts use the SFT adapter on TRAIN tasks only; resolved, loop-free, submitted trajectories are kept in
# the exact serving format (proxy logs), error turns masked. Nebius saw 11.4 -> 20.5% pass@1 from one round.

# %%
import os, subprocess, sys
CODE = "/kaggle/input/gemma-swe-code"
sys.path.insert(0, CODE)
from gemma_swe.kaggle_env import find_comp_dir, find_model_dir, find_wheel_dir, install_harness, pip_offline
COMP, DATA = find_comp_dir(), "/kaggle/input/gemma-swe-data"
CKPT = find_model_dir("gemma-4-e4b-it-qat-w4a16-ct")
ADAPTER = os.environ.get("ADAPTER", "/kaggle/input/gemma-swe-adapters/sft_v1")
SEEDS = os.environ.get("SEEDS", "0 1 2 3 4 5 6 7")
EVAL = "/kaggle/working/eval/rft_round1"

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

install_harness()
pip_offline(["peft", "compressed-tensors"], [find_wheel_dir("peft-*.whl")], no_deps=True)

# %% rollouts
sh(f"python {CODE}/scripts/build_submission.py --profile {CODE}/profiles/e4b_lora.yaml --adapter main_lora={ADAPTER} --dir /kaggle/working/sub_rollout")
sh(f"python {CODE}/scripts/local_eval.py --submission-dir /kaggle/working/sub_rollout --model-dir {CKPT} "
   f"--tasks {DATA}/tasks_train.jsonl --snapshots-dir {COMP}/snapshots --out {EVAL} --seeds {SEEDS} "
   f"--temperature 0.7 --concurrency 4")

# %% harvest
sh(f"python {CODE}/scripts/harvest_rft.py --eval-dir {EVAL} --tasks {DATA}/tasks_train.jsonl "
   f"--out /kaggle/working/rft_round1.jsonl --per-task 2")

# %% retrain: SFT data + RFT data (upweighted). Fresh adapter = cleaner ablation; --init-adapter = cheaper.
BF16 = "/kaggle/tmp/e4b-bf16"
if not os.path.exists(f"{BF16}/config.json"):
    sh(f"python {CODE}/scripts/dequantize_checkpoint.py --src {CKPT} --dst {BF16}")
sh(f"python {CODE}/scripts/build_sft_dataset.py --tokenizer {CKPT} --calibration {DATA}/calibration.json "
   f"--profile {CODE}/profiles/e4b_lora.yaml --source {DATA}/remapped.jsonl:1.0 --source {DATA}/graph_synth.jsonl:2.0 "
   f"--source /kaggle/working/rft_round1.jsonl:3.0 --max-len 16384 --out /kaggle/tmp/sft_rft1")
sh(f"torchrun --nproc_per_node 4 {CODE}/scripts/train_sft.py --model-dir {BF16} --data /kaggle/tmp/sft_rft1 "
   f"--out /kaggle/tmp/rft1_raw --rank 32 --alpha 32 --lr 1e-4 --epochs 1 --grad-accum 4 --ple-cpu")
sh(f"python {CODE}/scripts/check_adapter.py --adapter /kaggle/tmp/rft1_raw --base-ckpt {CKPT} --fix /kaggle/working/adapters/rft1")
# -> evaluate with NB04 (ADAPTER=/kaggle/input/gemma-swe-adapters/rft1) before promoting it.
