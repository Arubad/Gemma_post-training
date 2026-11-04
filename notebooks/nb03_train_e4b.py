# %% [markdown]
# # NB03 · LoRA SFT on Gemma 4 E4B (L4x4, internet OFF)
# **Inputs:** Kaggle Model e4b-it-qat-w4a16-ct, `gemma-swe-code`, `gemma-swe-data`, `gemma-swe-wheels`,
# competition wheelhouse · **GPU-h:** ~3-5 per epoch at 16k (x2 quota) · **Output:** `adapters/main_lora` + submission dir
#
# Recipe (report "Ranked recommendations" #3/#5/#6): harness-exact re-mapped trajectories + graph-path
# trajectories (oversampled), error-masked turns, length curriculum, LoRA r=32 alpha=32 on all 7 projections
# (KV-shared k/v skipped), lr 1e-4, 1 epoch, trained on the bf16 dequantization of the served checkpoint.
# Ablation arms for the paper: set `SOURCES` accordingly and give each run its own OUT (one account per arm).

# %%
import json, os, subprocess, sys
CODE = "/kaggle/input/gemma-swe-code"
sys.path.insert(0, CODE)
from gemma_swe.kaggle_env import find_model_dir, find_wheel_dir, gpu_summary, install_harness, pip_offline
DATA = "/kaggle/input/gemma-swe-data"
CKPT = find_model_dir("gemma-4-e4b-it-qat-w4a16-ct")
BF16 = "/kaggle/tmp/e4b-bf16"
RUN = os.environ.get("RUN", "sft_v1")
OUT = f"/kaggle/working/adapters/{RUN}"
SOURCES = [f"{DATA}/remapped.jsonl:1.0", f"{DATA}/graph_synth.jsonl:2.0"]  # + f"{DATA}/rft_round1.jsonl:2.0" in NB05
print(gpu_summary())

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

install_harness()  # brings a vLLM-compatible transformers
pip_offline(["peft", "compressed-tensors"], [find_wheel_dir("peft-*.whl")], no_deps=True)

# %% dequantize the served checkpoint (train on exactly the served int4 grid)
if not os.path.exists(f"{BF16}/config.json"):
    sh(f"python {CODE}/scripts/dequantize_checkpoint.py --src {CKPT} --dst {BF16}")

# %% build the tokenized dataset (prints labelled segments: eyeball them!)
srcs = " ".join(f"--source {s}" for s in SOURCES)
sh(f"python {CODE}/scripts/build_sft_dataset.py --tokenizer {CKPT} --calibration {DATA}/calibration.json "
   f"--profile {CODE}/profiles/e4b_lora.yaml {srcs} --max-len 16384 --out /kaggle/tmp/sft_{RUN} --workers 4 --show 2")

# %% smoke: 3 optimizer steps to check memory / NaNs / throughput before the full run
sh(f"torchrun --nproc_per_node 4 {CODE}/scripts/train_sft.py --model-dir {BF16} --data /kaggle/tmp/sft_{RUN} "
   f"--out /kaggle/tmp/smoke_adapter --max-steps 3 --grad-accum 2 --ple-cpu --log-steps 1")

# %% full run
sh(f"torchrun --nproc_per_node 4 {CODE}/scripts/train_sft.py --model-dir {BF16} --data /kaggle/tmp/sft_{RUN} "
   f"--out {OUT}_raw --rank 32 --alpha 32 --lr 1e-4 --epochs 1 --grad-accum 4 --ple-cpu --save-steps 100")

# %% repair keys against the SERVED checkpoint, validate, build the submission
sh(f"python {CODE}/scripts/check_adapter.py --adapter {OUT}_raw --base-ckpt {CKPT} --fix {OUT}")
sh(f"python {CODE}/scripts/build_submission.py --profile {CODE}/profiles/e4b_lora.yaml --adapter main_lora={OUT} "
   f"--base-ckpt {CKPT} --dir /kaggle/working/sub_{RUN} --zip /kaggle/working/submission_{RUN}.zip")
# -> Save Version; publish /kaggle/working/adapters as dataset `gemma-swe-adapters` for NB04/NB05.
