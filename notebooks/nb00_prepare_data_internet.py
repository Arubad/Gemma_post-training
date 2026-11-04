# %% [markdown]
# # NB00 · Prepare training data + offline wheels (CPU, INTERNET ON)
# **Accelerator:** None (CPU) · **Internet:** ON · **Inputs:** `gemma-swe-code` (this repo as a Dataset) · **GPU-h:** 0
#
# 1. Downloads the extra wheels the offline GPU notebooks need (PEFT, compressed-tensors, datasets...).
# 2. Streams public SWE-agent trajectories from Hugging Face and re-maps them to the 9 harness tools
#    (`gemma_swe.remap_openhands`): /workspace paths, <=150-line read_file chunks, scratch -> /tmp, no pip.
# 3. **Save Version**, then *New Dataset from output* -> name it `gemma-swe-data` (and `gemma-swe-wheels`).

# %%
import json, os, subprocess, sys
CODE = "/kaggle/input/gemma-swe-code"
sys.path.insert(0, CODE)
os.makedirs("/kaggle/working/data", exist_ok=True)
os.makedirs("/kaggle/working/wheels", exist_ok=True)

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

# %% [markdown]
# ## 1. Offline wheels for training (pin versions you validated; transformers must be > 5.5 for the Gemma 4 use_cache fix)

# %%
sh("pip download -q -d /kaggle/working/wheels --no-deps peft compressed-tensors safetensors")
sh("pip download -q -d /kaggle/working/wheels 'transformers>=5.6' accelerate datasets pyyaml")
sh("ls -la /kaggle/working/wheels | head -50")

# %% [markdown]
# ## 2. Re-map public trajectories (CC-BY-4.0 / MIT; attribute in the writeup)
# Use `--inspect` first if a dataset schema changed. Limits are raw rows scanned (streamed).

# %%
sh("pip install -q datasets pyyaml")
sh(f"python {CODE}/scripts/prepare_hf_trajectories.py --dataset nebius/SWE-rebench-openhands-trajectories:train:1 --out /tmp/x.jsonl --inspect")

# %%
sh(f"python {CODE}/scripts/prepare_hf_trajectories.py "
   "--dataset nebius/SWE-rebench-openhands-trajectories:train:40000 "
   "--dataset nvidia/SWE-Hero-openhands-trajectories:train:20000 "
   "--dataset SWE-bench/SWE-smith-trajectories:tool:15000 "
   "--out /kaggle/working/data/remapped.jsonl --max-calls 30 --per-instance 2")
print(open("/kaggle/working/data/remapped.stats.json").read())

# %% [markdown]
# Target: **2k-5k** records after filtering (report: gains appear at ~500 and grow log-linearly; LoRA at
# 16k context cannot use the 64-turn average trajectories). If fewer, raise the scan limits; if far more,
# keep the file - `build_sft_dataset.py --source remapped.jsonl:0.5` subsamples at build time.

# %%
sh("wc -l /kaggle/working/data/remapped.jsonl && head -c 1500 /kaggle/working/data/remapped.jsonl")
