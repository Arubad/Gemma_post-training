"""Quality filters for trajectories (both record formats)."""

from __future__ import annotations

import json
from collections import Counter

from .records import tool_call_sequence


def max_identical_run(seq: list[tuple[str, str]]) -> int:
    best = cur = 0
    prev = None
    for x in seq:
        cur = cur + 1 if x == prev else 1
        best = max(best, cur)
        prev = x
    return best


def max_duplicate_calls(seq: list[tuple[str, str]]) -> int:
    """Max number of times any (tool, args) pair appears anywhere (non-consecutive loops)."""
    if not seq:
        return 0
    return max(Counter(x for x in seq if x[0] not in ("get_status", "submit_patch")).values() or [0])


def n_budget_calls(seq: list[tuple[str, str]]) -> int:
    return sum(1 for n, _ in seq if n not in ("get_status", "submit_patch"))


def ends_with_submit(record: dict) -> bool:
    return any(n == "submit_patch" for n, _ in tool_call_sequence(record))


def missing_required_args(record: dict, required: dict[str, set[str]]) -> int:
    bad = 0
    for name, args in tool_call_sequence(record):
        a = json.loads(args)
        if name in required and not required[name].issubset(a):
            bad += 1
    return bad


def undeclared_calls(record: dict, declared: set[str]) -> int:
    return sum(1 for n, _ in tool_call_sequence(record) if n not in declared)


def passes(record: dict, max_calls: int = 30, max_repeat_run: int = 2, max_dupes: int = 3,
           required: dict[str, set[str]] | None = None, declared: set[str] | None = None) -> tuple[bool, str]:
    seq = tool_call_sequence(record)
    if not seq:
        return False, "no_tool_calls"
    if not ends_with_submit(record):
        return False, "no_submit"
    if n_budget_calls(seq) > max_calls:
        return False, "too_many_calls"
    if max_identical_run(seq) > max_repeat_run:
        return False, "loop"
    if max_duplicate_calls(seq) > max_dupes:
        return False, "repeated_calls"
    if declared and undeclared_calls(record, declared):
        return False, "undeclared_tool"
    if required and missing_required_args(record, required):
        return False, "missing_args"
    return True, "ok"
