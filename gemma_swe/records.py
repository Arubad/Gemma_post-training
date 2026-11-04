"""Trajectory record formats and assembly into the exact chat messages the scorer serves.

Two on-disk record formats (JSONL, one record per line):

1. "neutral" (re-mapped HF trajectories, graph-synthesized trajectories). Prompt-independent:
   {
     "id": str, "source": str, "repo": "owner/name", "problem_statement": str,
     "hints": str|None, "layout": [str]|None,
     "turns": [
        {"role": "assistant", "content": str, "tool_calls": [{"name": str, "arguments": {...}}], "train": bool},
        {"role": "tool", "name": str, "result": {...envelope...}},
        {"role": "user", "content": str},           # e.g. harness nudges
        ...
     ],
     "meta": {...}
   }
   System prompt, first user prompt, tool declarations and tool-content encoding are applied
   at assembly time from calibration.json + the deployed profile, so the same record can be
   re-rendered after the prompt changes.

2. "exact" (RFT trajectories harvested from proxy logs): the literal request messages + tools:
   {"id", "source", "format": "exact", "messages": [...], "tools": [...], "meta": {...}}
   Assistant messages may carry "train": false to mask them.
"""

from __future__ import annotations

import copy
import json
from typing import Iterable, Iterator

from .calibration import Calibration
from .harness_prompt import Budget


def read_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _args_dict(a) -> dict:
    if isinstance(a, dict):
        return a
    if a is None or a == "":
        return {}
    try:
        v = json.loads(a)
        return v if isinstance(v, dict) else {"value": v}
    except Exception:
        return {"value": a}


def assemble(
    record: dict,
    calib: Calibration,
    instruction: str,
    budget: Budget,
    agent_name: str = "coder",
    description: str | None = None,
    rationale_mode: str = "none",
) -> tuple[list[dict], list[dict]]:
    """-> (messages, tools). Assistant messages keep a boolean 'train' key (strip before templating).

    rationale_mode controls where a neutral record's per-step rationale goes:
      "none"     (default) dropped: tool-call turns have empty content. Gemma 4's template renders an
                 assistant message as tool_call -> tool_response -> content and closes the turn with
                 <turn|> when content is non-empty, so text before a call is re-rendered AFTER the result
                 and leaves the next serve-time prompt without a model-turn opener (the community's
                 "premature <turn|> -> loops" bug). Bare tool calls keep history == generation.
      "thinking" moved to reasoning_content (rendered as <|channel>thought ...<channel|> BEFORE the call);
                 train with enable_thinking=True and serve with a small thinking_budget.
      "content"  kept as content (only for templates that render content before calls).
    Final text-only answers always keep their content.
    """
    if rationale_mode not in ("none", "thinking", "content"):
        raise ValueError(f"bad rationale_mode {rationale_mode}")
    if record.get("format") == "exact":
        msgs = copy.deepcopy(record["messages"])
        for m in msgs:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    fn["arguments"] = _args_dict(fn.get("arguments"))
                m.setdefault("train", True)
                if m.get("content") is None:
                    m["content"] = ""
        return msgs, record.get("tools") or calib.get_tools()

    msgs: list[dict] = [
        {"role": "system", "content": calib.get_system(instruction, agent_name, description)},
        {
            "role": "user",
            "content": calib.get_user(
                record["repo"], record["problem_statement"], budget, record.get("layout"), record.get("hints")
            ),
        },
    ]
    call_idx = 0
    pending: list[str] = []
    for t in record["turns"]:
        role = t["role"]
        if role == "assistant":
            calls = []
            for tc in t.get("tool_calls") or []:
                cid = f"call_{call_idx}"
                call_idx += 1
                pending.append(cid)
                calls.append(
                    {"id": cid, "type": "function", "function": {"name": tc["name"], "arguments": _args_dict(tc.get("arguments"))}}
                )
            text = t.get("content") or ""
            m = {"role": "assistant", "content": text, "train": bool(t.get("train", True))}
            if calls:
                m["tool_calls"] = calls
                if rationale_mode == "none":
                    m["content"] = ""
                elif rationale_mode == "thinking":
                    m["content"] = ""
                    if text:
                        m["reasoning_content"] = text
            msgs.append(m)
        elif role == "tool":
            cid = pending.pop(0) if pending else f"call_{call_idx}"
            msgs.append(
                {"role": "tool", "tool_call_id": cid, "name": t["name"], "content": calib.wrap_tool_content(t["result"])}
            )
        elif role == "user":
            msgs.append({"role": "user", "content": t["content"]})
        else:
            raise ValueError(f"unknown role {role}")
    return msgs, calib.get_tools()


def strip_train_keys(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m2 = {k: v for k, v in m.items() if k != "train"}
        out.append(m2)
    return out


def tool_call_sequence(record: dict) -> list[tuple[str, str]]:
    """[(tool_name, canonical_args_json)] for loop detection, either format."""
    seq = []
    if record.get("format") == "exact":
        for m in record["messages"]:
            if m.get("role") == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function", {})
                    seq.append((fn.get("name", ""), json.dumps(_args_dict(fn.get("arguments")), sort_keys=True)))
    else:
        for t in record["turns"]:
            if t["role"] == "assistant":
                for tc in t.get("tool_calls") or []:
                    seq.append((tc["name"], json.dumps(_args_dict(tc.get("arguments")), sort_keys=True)))
    return seq
