"""Reconstruction of the harness's initial user prompt (build_agent_prompt) and the ADK
system prompt composition.

Exact strings should come from a capture (calibration.json). These reconstructions follow
HARNESS_README.md section 5.2 and are used (a) before a capture exists and (b) to fill in
per-task fields (repo, problem statement, layout) into a captured template.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .constants import COMMAND_TIMEOUT_S, MAX_FILE_CHARS, MAX_FILE_LINES, MAX_STDOUT_CHARS


@dataclass
class Budget:
    time_minutes: float = 5.0
    tool_calls: int | None = 40
    turns: int | None = 80
    command_timeout_s: int = 120

    @classmethod
    def from_eval_config(cls, ev: dict) -> "Budget":
        return cls(
            time_minutes=float(ev.get("max_time_minutes", 60.0)),
            tool_calls=ev.get("max_tool_calls"),
            turns=ev.get("max_turns"),
            command_timeout_s=int(ev.get("timeout_seconds", COMMAND_TIMEOUT_S)),
        )


def _fmt_minutes(m: float) -> str:
    return f"{float(m):.1f}"


def build_user_prompt(
    repo: str,
    problem: str,
    budget: Budget,
    layout: list[str] | None,
    hints: str | None = None,
    declared_tools: set[str] | None = None,
    graph_available: bool = True,
) -> str:
    """Reconstructed build_agent_prompt() (README 5.2, 7 sections)."""
    declared_tools = declared_tools or set()
    parts = [
        f"You are evaluating a software engineering task for repository {repo}.\n\n"
        f"Problem Statement:\n{problem}"
    ]
    if hints and hints.strip():
        parts.append(f"## Hints:\n{hints.strip()}")

    b = ["## Task Budget (Session terminates when any budget is exhausted)"]
    b.append(f"- Time allowance: {_fmt_minutes(budget.time_minutes)} minutes")
    if budget.tool_calls:
        b.append(f"- Tool calls allowance: {budget.tool_calls} calls")
    if budget.turns:
        b.append(f"- Max loop iterations: {budget.turns} turns")
    parts.append("\n".join(b))

    r = ["## Execution Environment Rules"]
    r.append(
        f"- Single command timeout: {budget.command_timeout_s} seconds "
        "(commands exceeding this fail without ending the session)"
    )
    r.append(f"- Command output limit: {MAX_STDOUT_CHARS} characters")
    if "read_file" in declared_tools or not declared_tools:
        r.append(f"- File view limit: {MAX_FILE_LINES} lines per read_file call")
        r.append(f"- File character limit: {MAX_FILE_CHARS} characters per read_file call")
    r.append(
        "- Environment is offline (no network/PyPI access). All repository and test "
        "dependencies are ALREADY pre-installed. Do NOT attempt to run pip install or "
        "download packages."
    )
    parts.append("\n".join(r))

    parts.append(
        "## Instructions\n"
        "0. Work strictly inside /workspace.\n"
        "1. Inspect the repository and its existing conventions before changing code.\n"
        "2. Implement the change required by the problem statement.\n"
        "3. Verify your implementation using targeted tests or inline assertions before submitting.\n"
        "4. Call submit_patch once your change is complete.\n"
        "5. Then return a short final text response."
    )

    g = sorted(t for t in declared_tools if t in {"search_similar_code", "get_code_neighbors", "get_code_subgraph"})
    if graph_available and g:
        lines = [
            "## Code Intelligence Tools",
            "This repository has pre-built code graph and embedding data indexed by dotted symbol "
            "path (e.g. `pkg.module.Class.method`). Use these tools for fast, targeted navigation:",
        ]
        if "search_similar_code" in g:
            lines.append(
                "- `search_similar_code(query)`: Find functions/classes with similar precomputed "
                "embeddings given a symbol name (e.g. `pkg.module.Class.method`, `Class.method`, "
                "or `func_name`)."
            )
        if "get_code_neighbors" in g:
            lines.append(
                "- `get_code_neighbors(node)`: Find callers and callees of a symbol node "
                "(e.g. `pkg.module.Class.method` or `Class.method`)."
            )
        if "get_code_subgraph" in g:
            lines.append(
                "- `get_code_subgraph(nodes)`: Get the induced call subgraph connecting a list of "
                "symbol nodes."
            )
        parts.append("\n".join(lines))

    if layout:
        parts.append("## Workspace Layout\n" + "\n".join(layout))
    return "\n\n".join(parts)


def adk_system_prompt(instruction: str, agent_name: str, description: str | None) -> str:
    """ADK composes: agent.instruction, then the identity instruction appended."""
    ident = f'You are an agent. Your internal name is "{agent_name}".'
    if description:
        ident += f' The description about you is "{description}".'
    return f"{instruction.rstrip()}\n\n{ident}"


# --------------------------------------------------------------------------------------
# Workspace layout (README 5.2 item 7)
# --------------------------------------------------------------------------------------
_PRUNE_BELOW_DEPTH1 = {"docs", "docs_src", "tests", "examples", "benchmarks"}


def _skip(name: str) -> bool:
    return (
        name.startswith(".")
        or name == "__pycache__"
        or name.endswith(".pyc")
        or name.endswith(".egg-info")
    )


def workspace_layout(root: str | os.PathLike, max_entries: int = 200, max_depth: int = 3) -> list[str]:
    """Breadth-first `find . -mindepth 1 -maxdepth 3` with the harness's pruning rules."""
    root = Path(root)
    entries: list[str] = []
    frontier: list[tuple[Path, int]] = [(root, 0)]
    while frontier and len(entries) < max_entries:
        nxt: list[tuple[Path, int]] = []
        for d, depth in frontier:
            try:
                children = sorted(d.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for c in children:
                if _skip(c.name):
                    continue
                rel = c.relative_to(root).as_posix()
                entries.append("./" + rel + ("/" if c.is_dir() else ""))
                if len(entries) >= max_entries:
                    break
                if c.is_dir() and depth + 1 < max_depth:
                    if depth >= 1 and c.name in _PRUNE_BELOW_DEPTH1:
                        continue
                    if depth == 0 and c.name in _PRUNE_BELOW_DEPTH1:
                        # keep the top-level entry but do not descend
                        continue
                    nxt.append((c, depth + 1))
            if len(entries) >= max_entries:
                break
        frontier = nxt
    return sorted(entries)


def layout_from_paths(paths: list[str], max_entries: int = 200, max_depth: int = 3) -> list[str]:
    """Approximate layout for re-mapped trajectories whose repo we do not have on disk."""
    seen: set[str] = set()
    for p in paths:
        p = p.strip().lstrip("./").lstrip("/")
        if not p or p.startswith("tmp/"):
            continue
        parts = p.split("/")
        for i in range(1, min(len(parts), max_depth) + 1):
            sub = "/".join(parts[:i])
            is_dir = i < len(parts)
            if any(_skip(x) for x in parts[:i]):
                break
            seen.add("./" + sub + ("/" if is_dir else ""))
    return sorted(seen)[:max_entries]
