"""Kaggle notebook helpers: locate inputs and install offline wheels.

Kaggle mount paths change between dataset/model versions (e.g. the model path segment
`/kaggle/input/gemma-4/<framework>/<variant>/<version>`), so everything is discovered by glob.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

INPUT = Path(os.environ.get("KAGGLE_INPUT", "/kaggle/input"))


def find_model_dir(variant: str) -> Path:
    """Directory containing config.json for e.g. 'gemma-4-e4b-it-qat-w4a16-ct' (case-insensitive)."""
    hits = []
    for cfg in glob.glob(str(INPUT / "**" / "config.json"), recursive=True):
        d = Path(cfg).parent
        if variant.lower() in str(d).lower() and any(d.glob("*.safetensors")):
            hits.append(d)
    if not hits:
        raise FileNotFoundError(f"no model dir matching {variant} under {INPUT}; attach it from Kaggle Models")
    return sorted(hits, key=lambda p: (len(str(p)), str(p)))[-1]


def find_comp_dir() -> Path:
    for c in (INPUT / "gemma-4-developer-agent", INPUT / "competitions" / "gemma-4-developer-agent"):
        if (c / "tasks.jsonl").exists():
            return c
    for t in glob.glob(str(INPUT / "**" / "tasks.jsonl"), recursive=True):
        p = Path(t).parent
        if (p / "snapshots").exists():
            return p
    raise FileNotFoundError("competition data not attached")


def find_wheel_dir(pattern: str) -> Path | None:
    hits = glob.glob(str(INPUT / "**" / pattern), recursive=True)
    return Path(hits[0]).parent if hits else None


def pip_offline(packages: list[str], wheel_dirs: list[Path | str], no_deps: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "-q", "--no-index"]
    for d in wheel_dirs:
        if d:
            cmd += ["--find-links", str(d)]
    if no_deps:
        cmd.append("--no-deps")
    cmd += packages
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def install_harness() -> None:
    """Install swegemma + adk-submission + adk-eval-core + google-adk (+ vLLM) from the wheelhouse."""
    wh = find_wheel_dir("swegemma-*.whl")
    if wh is None:
        raise FileNotFoundError("attach the competition wheelhouse dataset (contains swegemma-*.whl)")
    pkgs = ["swegemma", "adk-submission", "adk-eval-core", "google-adk"]
    if list(wh.glob("vllm-*.whl")):
        pkgs.append("vllm")
    pip_offline(pkgs, [wh])


def gpu_summary() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
                              capture_output=True, text=True).stdout
    except FileNotFoundError:
        return "no GPU"
