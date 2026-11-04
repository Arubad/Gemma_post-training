# %% [markdown]
# # NB01 · Capture the exact harness prompt + synthesize graph trajectories (CPU, internet OFF)
# **Accelerator:** None · **Internet:** OFF · **Inputs:** competition data, competition wheelhouse,
# `gemma-swe-code`, `gemma-swe-data` (from NB00) · **GPU-h:** 0
#
# * `calibration.json` = the literal system prompt, user prompt, tool declarations and tool-result encoding
#   ADK/LiteLLM send to vLLM (captured with a scripted mock model). **Every** training record is rendered
#   through it -> train == serve. Re-run this whenever `prompts/system.md` or the harness version changes.
# * Task split: train = fastapi + requests + httpx, held-out = all of `Textualize/rich` (unseen-repo proxy).
# * Hop map + graph-path trajectories (the paper contribution) from the train split only.

# %%
import json, os, shutil, subprocess, sys
CODE = "/kaggle/input/gemma-swe-code"
sys.path.insert(0, CODE)
from gemma_swe.kaggle_env import find_comp_dir, install_harness
OUT = "/kaggle/working/data"
os.makedirs(OUT, exist_ok=True)
COMP = find_comp_dir()
print("competition data:", COMP)

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

install_harness()
sh("swegemma eval --help | head -60")

# %%
sh(f"python {CODE}/scripts/split_tasks.py --tasks {COMP}/tasks.jsonl --out-dir {OUT}")

# %% [markdown]
# ## Capture (mock model on :8000, ~2 min). Inspect the printout: tool names, tool_content_style, user prompt head.

# %%
sh(f"python {CODE}/scripts/capture_harness_prompt.py --profile {CODE}/profiles/e4b_prompt.yaml "
   f"--tasks {OUT}/tasks_smoke.jsonl --snapshots-dir {COMP}/snapshots --out {OUT}/calibration.json")
cal = json.load(open(f"{OUT}/calibration.json"))
print(cal["tool_content_style"], [t["function"]["name"] for t in cal["tools"]])
print(cal["user_example"][:2500])

# %% [markdown]
# ## Hop-depth map for all 129 tasks (for hop-stratified evaluation) and graph-path SFT data (train split only)

# %%
sh(f"python {CODE}/scripts/hop_map.py --tasks {COMP}/tasks.jsonl --comp-dir {COMP} --out {OUT}/hop_map.json")
sh(f"python {CODE}/scripts/synth_graph_trajectories.py --tasks {OUT}/tasks_train.jsonl --comp-dir {COMP} "
   f"--out {OUT}/graph_synth.jsonl --workers 4 --workdir /kaggle/tmp")
print(open(f"{OUT}/graph_synth.stats.json").read())

# %%
# carry NB00's remapped data forward so later notebooks need a single data input
for f in ("remapped.jsonl", "remapped.stats.json"):
    src = f"/kaggle/input/gemma-swe-data/{f}"
    if os.path.exists(src):
        shutil.copy(src, OUT)
sh(f"ls -la {OUT}")
# -> Save Version, then update the `gemma-swe-data` dataset with this output.
