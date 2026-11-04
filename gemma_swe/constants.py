"""Shared constants mirrored from HARNESS_README.md (swegemma harness)."""

from __future__ import annotations

import os
import re
from pathlib import Path

# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------
MODELS = {
    "e2b": "gemma-4-e2b-it-qat-w4a16-ct",
    "e4b": "gemma-4-e4b-it-qat-w4a16-ct",
    "12b": "gemma-4-12b-it-qat-w4a16-ct",
    "31b": "gemma-4-31b-it-qat-w4a16-ct",
}
ALLOWED_MODEL_NAMES = set(MODELS.values())

# vLLM flags the scorer uses (scripts/inference.py -> setup_vllm_server)
VLLM_MAX_MODEL_LEN = 32768
VLLM_TP = 4
VLLM_GPU_UTIL = 0.90
VLLM_EXTRA_ARGS = {
    "gemma-4-12b-it-qat-w4a16-ct": [
        "--hf-overrides",
        '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
    ],
}

# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------
# Canonical order. The submission builder writes tools in this order and the training
# data declares them in this order, so the rendered system turn is identical.
TOOL_ORDER = [
    "run_command",
    "read_file",
    "edit_file",
    "write_file",
    "search_similar_code",
    "get_code_neighbors",
    "get_code_subgraph",
    "get_status",
    "submit_patch",
]
HARNESS_TOOLS = set(TOOL_ORDER)
FREE_TOOLS = {"submit_patch", "get_status"}  # do not count toward max_tool_calls
GRAPH_TOOLS = {"search_similar_code", "get_code_neighbors", "get_code_subgraph"}

# Harness limits (HarnessLimits)
MAX_STDOUT_CHARS = 5000
MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10000
COMMAND_TIMEOUT_S = 300

# Context compaction the evaluator configures (EventsCompactionConfig)
COMPACTION_TOKEN_THRESHOLD = 14336

# --------------------------------------------------------------------------------------
# Patch protection (strip_protected_files_from_patch)
# --------------------------------------------------------------------------------------
PROTECTED_CONFIG_FILES = {
    "conftest.py",
    "pytest.ini",
    "pyproject.toml",
    "tox.ini",
    "setup.cfg",
    ".pytest.ini",
    "sitecustomize.py",
    "usercustomize.py",
    "_swegemma_stubs.py",
}
_TEST_DIR_PARTS = {"tests", "test", "testing"}


def is_protected_path(path: str) -> bool:
    """True if the harness strips this path from agent patches."""
    p = path[2:] if path.startswith("./") else path
    p = p.lstrip("/")
    name = p.rsplit("/", 1)[-1]
    if name in PROTECTED_CONFIG_FILES or name.endswith(".pth"):
        return True
    if name.endswith(".py"):
        if name.startswith("test_") or name.endswith("_test.py"):
            return True
        parts = p.split("/")[:-1]
        if any(part in _TEST_DIR_PARTS for part in parts):
            return True
    return False


# --------------------------------------------------------------------------------------
# Submission limits (build_submission_limits)
# --------------------------------------------------------------------------------------
MAX_SUBMISSION_BYTES = 3 * 1024**3
MAX_FILE_COUNT = 10_000
ALLOWED_FILE_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MAX_LORA_RANK = 128
ROOT_CONFIG_NAMES = ("agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml")
MAX_OUTPUT_TOKENS_CAP = 32768

# --------------------------------------------------------------------------------------
# Paths (Kaggle). Every path can be overridden with an env var.
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent


def _first_existing(cands: list[str]) -> Path | None:
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


def comp_data_dir() -> Path | None:
    """Competition dataset root (tasks.jsonl, snapshots/, graphs/, embeddings/, wheels/...)."""
    return _first_existing(
        [
            os.environ.get("GSWE_COMP_DIR", ""),
            "/kaggle/input/gemma-4-developer-agent",
            "/kaggle/input/competitions/gemma-4-developer-agent",
            str(REPO_ROOT / "data"),
        ]
    )


def tasks_path() -> Path:
    env = os.environ.get("GSWE_TASKS")
    if env:
        return Path(env)
    d = comp_data_dir()
    if d is not None and (d / "tasks.jsonl").exists():
        return d / "tasks.jsonl"
    return REPO_ROOT / "tasks.jsonl"


# Held-out split used for every local evaluation: a whole repo the model never trains on,
# as a proxy for the hidden test set's private repos (see HANDOFF.md section 7.3).
HELDOUT_REPOS = {"Textualize/rich"}
