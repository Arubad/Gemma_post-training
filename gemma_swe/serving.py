"""Launch vLLM exactly like the scorer (scripts/inference.py -> setup_vllm_server) and drive
`swegemma eval` against it through the logging proxy."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from .adapter_utils import read_safetensors_header  # noqa: F401  (re-export for notebooks)
from .constants import VLLM_EXTRA_ARGS, VLLM_GPU_UTIL, VLLM_MAX_MODEL_LEN, VLLM_TP


def submission_adapters(sub_dir: str | Path) -> dict[str, str]:
    d = Path(sub_dir) / "adapters"
    if not d.exists():
        return {}
    return {p.name: str(p) for p in sorted(d.iterdir()) if (p / "adapter_config.json").exists()}


def adapter_rank(path: str | Path) -> int:
    return int(json.loads((Path(path) / "adapter_config.json").read_text()).get("r", 16))


_VLLM_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


def vllm_max_lora_rank(ranks) -> int:
    """vLLM only accepts specific --max-lora-rank values; round the largest adapter rank up."""
    r = max(ranks)
    return next(x for x in _VLLM_RANKS if x >= r)


def vllm_command(model_dir: str, served_name: str, adapters: dict[str, str] | None = None, port: int = 8001,
                 gpu_util: float = VLLM_GPU_UTIL, max_model_len: int = VLLM_MAX_MODEL_LEN, tp: int = VLLM_TP,
                 extra: list[str] | None = None) -> list[str]:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_dir, "--served-model-name", served_name,
        "--tensor-parallel-size", str(tp), "--gpu-memory-utilization", str(gpu_util),
        "--max-model-len", str(max_model_len),
        "--enable-auto-tool-choice", "--tool-call-parser", "gemma4", "--reasoning-parser", "gemma4",
        "--host", "127.0.0.1", "--port", str(port),
    ]
    if adapters:
        cmd += ["--enable-lora", "--max-loras", str(max(1, len(adapters))),
                "--max-lora-rank", str(vllm_max_lora_rank(adapter_rank(p) for p in adapters.values())), "--lora-modules"]
        cmd += [f"{n}={p}" for n, p in adapters.items()]
    cmd += VLLM_EXTRA_ARGS.get(served_name, [])
    cmd += extra or []
    return cmd


def start_vllm(cmd: list[str], log_path: str | Path, env: dict | None = None) -> subprocess.Popen:
    e = dict(os.environ)
    e.setdefault("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1")  # avoids LoRA OOM (thread 749191)
    e.update(env or {})
    lf = open(log_path, "w")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=e, start_new_session=True)


def wait_ready(base_url: str, timeout_s: int = 1800, proc: subprocess.Popen | None = None) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"vLLM exited with code {proc.returncode}; see its log")
        try:
            with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=5) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("vLLM did not become ready")


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    import signal

    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(60)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def eval_config(sub_dir: str | Path) -> dict:
    p = Path(sub_dir) / "eval_config.yaml"
    return (yaml.safe_load(p.read_text()) or {}).get("evaluation", {}) if p.exists() else {}


def swegemma_cmd(tasks: str, snapshots_dir: str, sub_dir: str, results_dir: str, concurrency: int = 1,
                 ev: dict | None = None, extra: list[str] | None = None, task_ids: list[str] | None = None) -> list[str]:
    exe = shutil.which("swegemma") or "swegemma"
    cmd = [exe, "eval", "--tasks", tasks, "--snapshots-dir", snapshots_dir, "--submission-dir", sub_dir,
           "--results-dir", results_dir, "--sandbox", "subprocess", "--concurrency", str(concurrency)]
    ev = ev or {}
    if ev.get("max_tool_calls"):
        cmd += ["--max-tool-calls", str(ev["max_tool_calls"])]
    if ev.get("max_time_minutes"):
        cmd += ["--max-time-minutes", str(ev["max_time_minutes"])]
    if ev.get("timeout_seconds"):
        cmd += ["--timeout-seconds", str(ev["timeout_seconds"])]
    if task_ids:
        cmd += ["--task-ids", *task_ids]
    cmd += extra or []
    return cmd


def set_seed(sub_dir: str | Path, seed: int, temperature: float | None = None) -> None:
    """Rewrite every configs/*sampling*.yaml (and sub_agents/*sampling*.yaml) seed in place."""
    for p in list(Path(sub_dir).glob("configs/*.yaml")) + list(Path(sub_dir).glob("sub_agents/*sampling*.yaml")):
        d = yaml.safe_load(p.read_text()) or {}
        if not isinstance(d, dict):
            continue
        d["seed"] = int(seed)
        if temperature is not None:
            d["temperature"] = float(temperature)
        p.write_text(yaml.safe_dump(d, sort_keys=False))
