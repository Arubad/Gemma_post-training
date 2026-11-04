"""OpenAI-style declarations of the 9 harness tools.

These are a *reconstruction* from HARNESS_README.md section 6. The exact JSON that ADK/LiteLLM
sends to vLLM (descriptions are the swegemma docstrings) should be captured once with
`scripts/capture_harness_prompt.py`; the captured list is stored in calibration.json and takes
precedence everywhere (see calibration.Calibration.tools). Training on the captured
declarations makes the rendered system turn byte-identical to what the scorer serves.
"""

from __future__ import annotations

import copy

from .constants import TOOL_ORDER


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_DEFAULT_TOOLS: dict[str, dict] = {
    "run_command": _fn(
        "run_command",
        "Executes a shell command in /bin/bash -c inside /workspace.",
        {"command": {"type": "string"}},
        ["command"],
    ),
    "submit_patch": _fn(
        "submit_patch",
        "Stages untracked file intents (git add -N .) and captures git diff HEAD from /workspace.",
        {},
        [],
    ),
    "get_status": _fn(
        "get_status",
        "Returns live budget consumption and patch status.",
        {},
        [],
    ),
    "read_file": _fn(
        "read_file",
        "Reads a file from /workspace with 1-indexed inclusive line slicing.",
        {
            "filepath": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"},
        },
        ["filepath"],
    ),
    "edit_file": _fn(
        "edit_file",
        "Replaces old_string with new_string in an existing non-empty file inside /workspace.",
        {
            "filepath": {"type": "string"},
            "old_string": {"type": "string"},
            "new_string": {"type": "string"},
            "allow_multiple": {"type": "boolean"},
        },
        ["filepath", "old_string", "new_string"],
    ),
    "write_file": _fn(
        "write_file",
        "Creates or overwrites a file at /workspace/<filepath>, automatically creating parent "
        "directories (mkdir -p).",
        {"filepath": {"type": "string"}, "content": {"type": "string"}},
        ["filepath", "content"],
    ),
    "get_code_neighbors": _fn(
        "get_code_neighbors",
        "Finds incoming and outgoing neighbors of a symbol (node) in the repository "
        "call/dependency graph.",
        {
            "node": {"type": "string"},
            "edge_type": {"type": "string"},
            "max_neighbors": {"type": "integer"},
        },
        ["node"],
    ),
    "search_similar_code": _fn(
        "search_similar_code",
        "Finds top-k graph nodes with highest cosine similarity to query in the pre-computed "
        "embeddings.",
        {"query": {"type": "string"}, "k": {"type": "integer"}},
        ["query"],
    ),
    "get_code_subgraph": _fn(
        "get_code_subgraph",
        "Extracts the induced subgraph (all nodes and interconnecting edges) for a list of symbols.",
        {"nodes": {"type": "array", "items": {"type": "string"}}},
        ["nodes"],
    ),
}


def default_tools(names: list[str] | None = None) -> list[dict]:
    """Tool declarations in canonical order (optionally restricted to `names`)."""
    names = names or TOOL_ORDER
    order = [n for n in TOOL_ORDER if n in names]
    return [copy.deepcopy(_DEFAULT_TOOLS[n]) for n in order]


def tool_names(tools: list[dict]) -> list[str]:
    out = []
    for t in tools:
        fn = t.get("function", t)
        out.append(fn["name"])
    return out


def required_args(tools: list[dict]) -> dict[str, set[str]]:
    out = {}
    for t in tools:
        fn = t.get("function", t)
        out[fn["name"]] = set(fn.get("parameters", {}).get("required", []))
    return out
