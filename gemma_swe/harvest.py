"""Turn proxy logs + swegemma results into exact-format SFT records (rejection-sampling FT).

Grouping: requests are keyed by (system prompt, first user message). Within a conversation the
request at step k+1 normally extends request k by [assistant reply k, tool results ...]. If ADK
compaction rewrites history, the extension breaks; we then start a new chain. Each chain's last
request + its reply is one training sample in which every assistant turn is labelled (masked
where a turn produced a harness error, or was a nudge-induced text-only turn).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .records import read_jsonl

NON_TRAIN_ERRORS = {"FileEditError", "ValidationError", "ToolNotFound", "ValueError", "TypeError", "KeyError"}
NUDGE_PREFIXES = (
    "Your previous response reached the token limit",
    "Please continue your work using the available tools",
)


def _text(c) -> str:
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return c or ""


def conv_key(messages: list[dict]) -> str:
    sys_ = next((_text(m.get("content")) for m in messages if m.get("role") in ("system", "developer")), "")
    usr = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    return hashlib.sha1((sys_ + "\x00" + usr).encode()).hexdigest()[:16]


def _norm(m: dict) -> str:
    """Comparable form of a message (arguments normalised, ids ignored)."""
    d = {"role": m.get("role"), "content": _text(m.get("content"))}
    if m.get("tool_calls"):
        calls = []
        for tc in m["tool_calls"]:
            fn = tc.get("function", {})
            a = fn.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    pass
            calls.append([fn.get("name"), a])
        d["tool_calls"] = calls
    if m.get("role") == "tool":
        d["name"] = m.get("name")
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _reply_msg(resp: dict) -> dict | None:
    try:
        m = dict(resp["choices"][0]["message"])
    except (KeyError, IndexError, TypeError):
        return None
    m.pop("reasoning_content", None)  # thinking off in training (see chat_render)
    m.pop("reasoning", None)
    m["role"] = "assistant"
    return m


def conversations(log_path: str | Path) -> dict[str, list[dict]]:
    convs: dict[str, list[dict]] = defaultdict(list)
    for rec in read_jsonl(log_path):
        req = rec.get("request") or {}
        if "error" in rec or not req.get("messages"):
            continue
        convs[conv_key(req["messages"])].append(rec)
    for k in convs:
        convs[k].sort(key=lambda r: r["ts"])
    return convs


def chains(recs: list[dict]) -> list[dict]:
    """Split one conversation's requests into maximal history-consistent chains.
    Returns [{"messages": full_messages_incl_final_reply, "tools": [...], "n_requests": int}]."""
    out = []
    cur_msgs: list[dict] | None = None
    cur_tools = None
    n = 0
    for r in recs:
        msgs = r["request"]["messages"]
        reply = _reply_msg(r.get("response") or {})
        if cur_msgs is not None:
            a = [_norm(m) for m in cur_msgs]
            b = [_norm(m) for m in msgs[: len(cur_msgs)]]
            if len(msgs) < len(cur_msgs) or a != b:
                out.append({"messages": cur_msgs, "tools": cur_tools, "n_requests": n})
                n = 0
        cur_msgs = list(msgs) + ([reply] if reply else [])
        cur_tools = r["request"].get("tools")
        n += 1
    if cur_msgs is not None:
        out.append({"messages": cur_msgs, "tools": cur_tools, "n_requests": n})
    return out


def _tool_error_type(content: str) -> str | None:
    try:
        d = json.loads(content)
        if isinstance(d, dict) and set(d) == {"result"} and isinstance(d["result"], str):
            d = json.loads(d["result"])
        if isinstance(d, dict) and d.get("status") == "error":
            return d.get("error_type") or "error"
    except Exception:
        return None
    return None


def mark_train_flags(messages: list[dict]) -> list[dict]:
    """Mask assistant turns whose tool call errored for format reasons, and nudged text turns."""
    msgs = [dict(m) for m in messages]
    for i, m in enumerate(msgs):
        if m.get("role") != "assistant":
            continue
        train = True
        if m.get("tool_calls"):
            ids = {tc.get("id") for tc in m["tool_calls"]}
            for nxt in msgs[i + 1 :]:
                if nxt.get("role") != "tool":
                    break
                if nxt.get("tool_call_id") in ids or not nxt.get("tool_call_id"):
                    et = _tool_error_type(_text(nxt.get("content")))
                    if et in NON_TRAIN_ERRORS:
                        train = False
        else:
            # text-only turn that is followed by a nudge = the model stalled
            nxt = msgs[i + 1] if i + 1 < len(msgs) else None
            if nxt and nxt.get("role") == "user" and _text(nxt.get("content")).startswith(NUDGE_PREFIXES):
                train = False
        m["train"] = train
    return msgs


def instance_for(messages: list[dict], tasks: list[dict]) -> str | None:
    usr = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    best = None
    for t in tasks:
        ps = (t.get("problem_statement") or "").strip()
        if ps and ps[:400] in usr:
            if best is None or len(ps) > len(best["problem_statement"]):
                best = t
    return best["instance_id"] if best else None


def load_results(results_dir: str | Path) -> dict[str, dict]:
    p = Path(results_dir) / "task_results.jsonl"
    out = {}
    if p.exists():
        for r in read_jsonl(p):
            iid = r.get("instance_id") or r.get("task_id") or r.get("id")
            if iid:
                out[iid] = r
    return out


def is_resolved(r: dict) -> bool:
    for k in ("resolved", "is_resolved", "success", "passed"):
        if k in r:
            return bool(r[k])
    return float(r.get("score", 0) or 0) >= 1.0


def harvest(log_path: str | Path, results_dir: str | Path, tasks: list[dict], run_tag: str,
            only_resolved: bool = True) -> list[dict]:
    results = load_results(results_dir)
    out = []
    for key, recs in conversations(log_path).items():
        chs = chains(recs)
        iid = instance_for(recs[0]["request"]["messages"], tasks)
        res = results.get(iid or "", {})
        resolved = is_resolved(res) if res else False
        if only_resolved and not resolved:
            continue
        for ci, ch in enumerate(chs):
            msgs = mark_train_flags(ch["messages"])
            out.append({
                "id": f"{run_tag}:{iid}:{key}:{ci}",
                "source": "rft",
                "format": "exact",
                "messages": msgs,
                "tools": ch["tools"],
                "meta": {"instance_id": iid, "resolved": resolved, "chain": ci, "n_chains": len(chs),
                         "n_requests": ch["n_requests"], "run": run_tag},
            })
    return out
