"""Emulated harness tool results (JSON envelopes, README section 6).

Used to render synthesized / re-mapped trajectories with observations that look like the
real tools' output (same keys, same truncation caps). RFT data harvested from real runs does
not use this module: it already contains the true observations.
"""

from __future__ import annotations

import difflib

from .constants import MAX_FILE_CHARS, MAX_FILE_LINES, MAX_STDOUT_CHARS


def _trunc(s: str, n: int = MAX_STDOUT_CHARS) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n]


def run_command(stdout: str, stderr: str = "", exit_code: int = 0, timed_out: bool = False,
                timeout_s: int = 300) -> dict:
    if timed_out:
        return {
            "status": "error",
            "error_type": "TimeoutExceeded",
            "error_message": f"Command timed out after {timeout_s} seconds",
            "details": {"stdout": _trunc(stdout), "stderr": _trunc(stderr)},
        }
    if exit_code == 0:
        return {"status": "ok", "stdout": _trunc(stdout), "stderr": _trunc(stderr), "exit_code": 0}
    return {
        "status": "error",
        "error_type": "CommandError",
        "error_message": f"Command exited with code {exit_code}",
        "details": {"stdout": _trunc(stdout), "stderr": _trunc(stderr), "exit_code": exit_code},
    }


def read_file(filepath: str, text: str, start_line: int | None = None, end_line: int | None = None) -> dict:
    lines = text.splitlines(keepends=True)
    total = len(lines)
    start = max(1, int(start_line or 1))
    end = total if end_line is None else min(int(end_line), total)
    if start > max(total, 1):
        return error("ValidationError", f"start_line {start} exceeds total lines {total}")
    out, chars, last, truncated = [], 0, start - 1, False
    for i in range(start, end + 1):
        if len(out) >= MAX_FILE_LINES:
            truncated = True
            break
        ln = lines[i - 1]
        if chars + len(ln) > MAX_FILE_CHARS:
            truncated = True
            break
        out.append(ln)
        chars += len(ln)
        last = i
    return {
        "status": "ok",
        "filepath": filepath,
        "content": "".join(out),
        "start_line": start,
        "end_line": last,
        "total_lines": total,
        "is_truncated": truncated,
    }


def unified_diff(filepath: str, before: str, after: str) -> str:
    d = difflib.unified_diff(
        before.splitlines(keepends=True),
        after.splitlines(keepends=True),
        fromfile=f"a/{filepath}",
        tofile=f"b/{filepath}",
    )
    return "".join(d)


def edit_file(filepath: str, before: str, after: str, occurrences: int = 1, strategy: str = "exact") -> dict:
    diff = unified_diff(filepath, before, after)
    return {
        "status": "ok",
        "filepath": filepath,
        "occurrences": occurrences,
        "strategy": strategy,
        "diff": _trunc(diff),
        "is_truncated": len(diff) > MAX_STDOUT_CHARS,
    }


def write_file(filepath: str, content: str) -> dict:
    return {"status": "ok", "filepath": filepath, "size": len(content.encode("utf-8"))}


def submit_patch(patch: str) -> dict:
    files = sum(1 for ln in patch.splitlines() if ln.startswith("+++ "))
    return {"status": "ok", "patch_size": len(patch.encode("utf-8")), "files_changed": files}


def get_status(used: int, max_calls: int, elapsed_s: float, max_minutes: float, patch_size: int = 0,
               submitted: bool = False, max_turns: int = 80, cmd_timeout: int = 120) -> dict:
    return {
        "status": "ok",
        "tool_calls_used": used,
        "patch_submitted": submitted,
        "patch_size": patch_size,
        "tool_calls_remaining": max(0, max_calls - used),
        "max_tool_calls": max_calls,
        "time_seconds_remaining": round(max(0.0, max_minutes * 60 - elapsed_s), 1),
        "max_time_minutes": float(max_minutes),
        "agent_elapsed_seconds": round(elapsed_s, 1),
        "max_turns": max_turns,
        "command_timeout_seconds": cmd_timeout,
    }


def neighbors(node: str, names: list[str], max_chars: int = MAX_STDOUT_CHARS) -> dict:
    out, used, trunc = [], 0, False
    for n in names:
        if used + len(n) > max_chars:
            trunc = True
            break
        out.append(n)
        used += len(n)
    return {"status": "ok", "node": node, "neighbors": out, "count": len(names), "is_truncated": trunc}


def search_similar(query: str, results: list[tuple[str, str, float]], max_chars: int = MAX_STDOUT_CHARS) -> dict:
    """results: (node_name, code, similarity). Character budget split across results with rollover."""
    k = max(1, len(results))
    budget = max_chars
    out, trunc = [], False
    remaining = list(results)
    for idx, (name, code, sim) in enumerate(remaining):
        share = budget // max(1, k - idx)
        snippet = code if len(code) <= share else code[:share]
        trunc = trunc or len(snippet) < len(code)
        budget -= len(snippet)
        out.append({"node_name": name, "code": snippet, "similarity": round(float(sim), 4)})
    return {"status": "ok", "query": query, "results": out, "count": len(out), "is_truncated": trunc}


def subgraph(nodes: list[str], edges: list[tuple[str, str, str]], max_chars: int = MAX_STDOUT_CHARS) -> dict:
    return {
        "status": "ok",
        "nodes": nodes,
        "edges": [{"from": a, "to": b, "type": t} for a, b, t in edges],
        "node_count": len(nodes),
        "edge_count": len(edges),
        "is_truncated": False,
    }


def error(error_type: str, message: str, details: dict | None = None) -> dict:
    d = {"status": "error", "error_type": error_type, "error_message": message}
    if details:
        d["details"] = details
    return d
