"""Deterministic re-mapping of public SWE-agent trajectories into the 9-tool harness schema.

Supported sources (tool names are aliased, so most OpenHands / SWE-agent tool-format datasets work):
  * nebius/SWE-rebench-openhands-trajectories, nvidia/SWE-Hero-openhands-trajectories,
    nvidia/Nemotron-SWE-v1 (OpenHands: execute_bash, str_replace_editor, think, finish)
  * SWE-bench/SWE-smith-trajectories "tool" split (bash, str_replace_editor, submit)

Output: neutral records (see records.py). Behavioural rewrites that make the data match *this*
harness rather than OpenHands:
  * repo root (/workspace/<dir>, /testbed, ...) -> /workspace (cwd of run_command); file tools
    get repo-relative paths
  * `view` of a file -> read_file chunks of <=150 lines (the harness cap); a view larger than
    `max_view_chunks*150` lines drops the trajectory (so the model never "knows" unseen lines)
  * `view` of a directory -> run_command find
  * scratch scripts created at the repo root (reproduce_*.py, debug_*.py, ...) -> written to /tmp
    via a run_command heredoc, and later references rewritten to /tmp/<name> (write_file can only
    write under /workspace and every untracked file there leaks into the patch)
  * `pip install` steps are removed (sandbox is offline)
  * think-tool thoughts and long assistant prose -> a 1-2 sentence rationale in `content`
  * failed edits (old_str not found / not unique) stay in context but are loss-masked
  * finish/submit -> submit_patch + a short final answer
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import observations as obs
from .harness_prompt import layout_from_paths

BASH_TOOLS = {"execute_bash", "bash", "run", "execute_command", "run_command", "terminal", "execute_ipython_cell_bash"}
EDITOR_TOOLS = {"str_replace_editor", "str_replace_based_edit_tool", "file_editor", "edit"}
FINISH_TOOLS = {"finish", "submit", "submit_patch"}
THINK_TOOLS = {"think"}
IGNORE_TOOLS = {"task_tracker", "browser", "web_read", "fetch"}

SCRATCH_RE = re.compile(r"^(reproduce|repro|debug|check|verify|test_issue|test_fix|test_repro|scratch|demo|example_issue)[\w\-]*\.py$")

_CWD_LINE = re.compile(r"^\[(Current working directory|Python interpreter|Command finished|The command completed|Command exited)[^\]]*\]\s*$", re.M)
_EXIT_RE = re.compile(r"exit code (-?\d+)")
_CAT_N = re.compile(r"^\s*(\d+)\t(.*)$")


class Drop(Exception):
    """Trajectory cannot be faithfully re-mapped."""


@dataclass
class RemapConfig:
    max_tool_calls: int = 30
    max_view_chunks: int = 3
    rationale_chars: int = 280
    require_resolved: bool = True
    require_edit: bool = True


@dataclass
class _State:
    root: str
    scratch: dict[str, str] = field(default_factory=dict)  # rel path -> /tmp path
    files_seen: dict[str, dict[int, str]] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    edits: int = 0


# ------------------------------------------------------------------------------------ helpers
def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _messages(rec: dict) -> list[dict]:
    for k in ("trajectory", "messages", "conversation", "conversations", "history"):
        v = rec.get(k)
        if v is None:
            continue
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, list) and v:
            return v
    raise Drop("no message list")


def _problem(rec: dict, msgs: list[dict]) -> str:
    for k in ("problem_statement", "issue", "task"):
        if isinstance(rec.get(k), str) and rec[k].strip():
            return rec[k].strip()
    user = next((_text(m.get("content")) for m in msgs if m.get("role") == "user"), "")
    for tag in ("issue_description", "pr_description", "issue", "problem_statement"):
        m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", user, re.S)
        if m:
            return m.group(1).strip()
    if user.strip():
        return user.strip()
    raise Drop("no problem statement")


def _repo(rec: dict) -> str:
    if isinstance(rec.get("repo"), str) and "/" in rec["repo"]:
        return rec["repo"]
    iid = rec.get("instance_id", "")
    m = re.match(r"^([\w.\-]+)__([\w.\-]+?)(?:-\d+|\.\w+)?$", iid)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return rec.get("repo") or "unknown/unknown"


def _detect_root(msgs: list[dict]) -> str:
    blob = "\n".join(_text(m.get("content")) for m in msgs[:3])
    m = re.search(r"<uploaded_files>\s*(/\S+?)\s*</uploaded_files>", blob)
    if m:
        return m.group(1).rstrip("/")
    allargs = " ".join(
        str(tc.get("function", {}).get("arguments", "")) for m in msgs for tc in (m.get("tool_calls") or [])
    )
    for cand in ("/testbed",):
        if cand in allargs:
            return cand
    m = re.search(r"(/workspace/[\w.\-]+)", allargs)
    if m:
        return m.group(1)
    return "/workspace"


def _args(tc: dict) -> dict:
    fn = tc.get("function", tc)
    a = fn.get("arguments", {})
    if isinstance(a, str):
        try:
            a = json.loads(a) if a.strip() else {}
        except json.JSONDecodeError as e:
            raise Drop(f"bad tool arguments: {e}")
    return a if isinstance(a, dict) else {}


def _rel(st: _State, path: str) -> str | None:
    """Repo-relative path, or None if outside the repo."""
    p = (path or "").strip()
    if p.startswith(st.root + "/"):
        p = p[len(st.root) + 1 :]
    elif p == st.root:
        p = "."
    elif p.startswith("/"):
        return None
    while p.startswith("./"):
        p = p[2:]
    return p or "."


def _cmd(st: _State, cmd: str) -> str:
    out = cmd.replace(st.root, "/workspace") if st.root != "/workspace" else cmd
    for rel, tmp in st.scratch.items():
        out = re.sub(rf"(?<![\w/.]){re.escape('/workspace/' + rel)}(?![\w/])", tmp, out)
        out = re.sub(rf"(?<![\w/.]){re.escape(rel)}(?![\w/])", tmp, out)
    return out


def _clean_output(st: _State, text: str) -> tuple[str, int]:
    code = 0
    m = None
    for m in _EXIT_RE.finditer(text):
        pass
    if m:
        code = int(m.group(1))
    text = _CWD_LINE.sub("", text).rstrip()
    if st.root != "/workspace":
        text = text.replace(st.root, "/workspace")
    return text, code


def _rationale(text: str, limit: int) -> str:
    t = re.sub(r"<[^>]{1,40}>", " ", text or "")
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", t)
    out = ""
    for s in sents[:2]:
        if len(out) + len(s) + 1 > limit:
            break
        out = f"{out} {s}".strip()
    return out or t[:limit].rsplit(" ", 1)[0]


def _parse_cat_n(text: str) -> list[tuple[int, str]]:
    rows = []
    for ln in text.splitlines():
        m = _CAT_N.match(ln)
        if m:
            rows.append((int(m.group(1)), m.group(2)))
    return rows


# ------------------------------------------------------------------------------------ main
def remap(rec: dict, cfg: RemapConfig | None = None) -> dict:
    cfg = cfg or RemapConfig()
    if cfg.require_resolved:
        r = rec.get("resolved", rec.get("success", rec.get("is_resolved")))
        if r is not None and not bool(r):
            raise Drop("not resolved")
    msgs = _messages(rec)
    problem = _problem(rec, msgs)
    st = _State(root=_detect_root(msgs))

    # tool results by call id
    results: dict[str, str] = {}
    for m in msgs:
        if m.get("role") in ("tool", "function") and m.get("tool_call_id"):
            results[m["tool_call_id"]] = _text(m.get("content"))

    turns: list[dict] = []
    pending_thought = ""
    n_calls = 0
    finished = False
    final_patch = rec.get("model_patch") or rec.get("patch") or rec.get("git_patch") or ""

    def add(content, name, args, result, train=True):
        nonlocal n_calls
        turns.append({"role": "assistant", "content": content, "tool_calls": [{"name": name, "arguments": args}], "train": train})
        turns.append({"role": "tool", "name": name, "result": result})
        if name not in ("submit_patch", "get_status"):
            n_calls += 1

    for m in msgs:
        if m.get("role") != "assistant" or finished:
            continue
        calls = m.get("tool_calls") or []
        prose = _rationale(_text(m.get("content")), cfg.rationale_chars)
        if not calls:
            # text-only assistant turn mid-trajectory: fold into next rationale
            pending_thought = prose or pending_thought
            continue
        for ci, tc in enumerate(calls):
            fn = tc.get("function", tc)
            name = fn.get("name", "")
            args = _args(tc)
            out_text = results.get(tc.get("id", ""), "")
            content = (prose if ci == 0 else "") or pending_thought
            if name in THINK_TOOLS:
                pending_thought = _rationale(args.get("thought", "") or args.get("content", ""), cfg.rationale_chars)
                continue
            if name in IGNORE_TOOLS:
                continue
            pending_thought = ""
            if name in BASH_TOOLS:
                cmd = (args.get("command") or args.get("cmd") or "").strip()
                if args.get("is_input") in (True, "true") or not cmd:
                    raise Drop("interactive bash input")
                if re.search(r"\bpip3?\s+install\b|\buv\s+pip\b|\bconda\s+install\b", cmd):
                    continue
                cmd = _cmd(st, cmd)
                text, code = _clean_output(st, out_text)
                timed_out = "timed out" in text.lower() and code in (-1, 124)
                add(content, "run_command", {"command": cmd}, obs.run_command(text, "", code, timed_out))
            elif name in EDITOR_TOOLS:
                _editor(st, cfg, args, out_text, content, add)
            elif name in FINISH_TOOLS:
                if not st.edits and cfg.require_edit:
                    raise Drop("finished without edits")
                patch = final_patch or "x" * 400
                add(content or "The change is implemented and verified; submitting the patch.", "submit_patch", {}, obs.submit_patch(patch))
                summary = _rationale(args.get("message", "") or args.get("summary", ""), cfg.rationale_chars)
                turns.append({"role": "assistant", "content": summary or "I have submitted the patch.", "train": True})
                finished = True
                break
            else:
                raise Drop(f"unsupported tool {name}")
            if n_calls > cfg.max_tool_calls:
                raise Drop("too many tool calls")
    if not finished:
        raise Drop("never finished/submitted")
    if cfg.require_edit and st.edits == 0:
        raise Drop("no edits")
    return {
        "id": str(rec.get("trajectory_id") or rec.get("instance_id") or rec.get("id") or ""),
        "source": rec.get("_source", "openhands"),
        "repo": _repo(rec),
        "problem_statement": problem,
        "hints": None,
        "layout": layout_from_paths(st.paths) or None,
        "turns": turns,
        "meta": {"n_tool_calls": n_calls, "root": st.root, "instance_id": rec.get("instance_id")},
    }


def _editor(st: _State, cfg: RemapConfig, args: dict, out_text: str, content: str, add) -> None:
    command = args.get("command", "")
    path = args.get("path", "")
    rel = _rel(st, path)
    if command == "view":
        is_dir = "files and directories" in out_text[:300] or (rel is not None and "." not in rel.rsplit("/", 1)[-1] and not _parse_cat_n(out_text))
        if is_dir:
            target = "/workspace" if rel in (None, ".") else f"/workspace/{rel}"
            listing, _ = _clean_output(st, out_text)
            listing = "\n".join(ln for ln in listing.splitlines() if not ln.startswith("Here's the files"))
            add(content, "run_command", {"command": f"find {target} -maxdepth 2 -not -path '*/.*' | sort | head -100"},
                obs.run_command(listing.strip() + "\n"))
            return
        rows = _parse_cat_n(out_text)
        if rel is None:
            # file outside repo (e.g. /tmp): plain cat
            body = "\n".join(t for _, t in rows) if rows else _clean_output(st, out_text)[0]
            add(content, "run_command", {"command": f"cat {path}"}, obs.run_command(body + "\n"))
            return
        if not rows:
            raise Drop("unparseable view output")
        st.paths.append(rel)
        seen = st.files_seen.setdefault(rel, {})
        for ln, t in rows:
            seen[ln] = t
        last = rows[-1][0]
        total = last if not args.get("view_range") else max(last, max(seen))
        chunks: list[list[tuple[int, str]]] = [[]]
        chars = 0
        for row in rows:  # respect both harness caps: 150 lines AND 10k chars per read_file
            ln_len = len(row[1]) + 1
            if chunks[-1] and (len(chunks[-1]) >= 150 or chars + ln_len > 10000):
                chunks.append([])
                chars = 0
            chunks[-1].append(row)
            chars += ln_len
        if len(chunks) > cfg.max_view_chunks:
            raise Drop("view larger than allowed read_file chunks")
        for c, chunk in enumerate(chunks):
            text = "".join(t + "\n" for _, t in chunk)
            s, e = chunk[0][0], chunk[-1][0]
            env = {"status": "ok", "filepath": rel, "content": text, "start_line": s, "end_line": e,
                   "total_lines": total, "is_truncated": False}
            a = {"filepath": rel}
            if s != 1 or args.get("view_range") or len(chunks) > 1:
                a.update({"start_line": s, "end_line": e})
            add(content if c == 0 else "Continuing to read the file.", "read_file", a, env)
        return
    if command == "create":
        text = args.get("file_text", "")
        if rel is None:
            tmp = path if path.startswith("/tmp/") else f"/tmp/{path.rsplit('/', 1)[-1]}"
            add(content, "run_command", {"command": f"cat > {tmp} << 'PYEOF'\n{text.rstrip()}\nPYEOF"}, obs.run_command(""))
            return
        base = rel.rsplit("/", 1)[-1]
        if "/" not in rel and SCRATCH_RE.match(base):
            tmp = f"/tmp/{base}"
            st.scratch[rel] = tmp
            add(content, "run_command", {"command": f"cat > {tmp} << 'PYEOF'\n{text.rstrip()}\nPYEOF"}, obs.run_command(""))
            return
        st.paths.append(rel)
        st.edits += 1
        add(content, "write_file", {"filepath": rel, "content": text}, obs.write_file(rel, text))
        return
    if command == "str_replace":
        if rel is None or rel in st.scratch:
            raise Drop("edit outside repo / on scratch file")
        old, new = args.get("old_str", ""), args.get("new_str", "") or ""
        low = out_text.lower()
        if "no replacement was performed" in low or "did not appear verbatim" in low:
            add(content, "edit_file", {"filepath": rel, "old_string": old, "new_string": new},
                obs.error("FileEditError", f"old_string not found in {rel}"), train=False)
            return
        if "multiple occurrences" in low:
            add(content, "edit_file", {"filepath": rel, "old_string": old, "new_string": new},
                obs.error("FileEditError", f"old_string matches multiple occurrences in {rel}; set allow_multiple=true or add context"), train=False)
            return
        st.paths.append(rel)
        st.edits += 1
        add(content, "edit_file", {"filepath": rel, "old_string": old, "new_string": new}, obs.edit_file(rel, old + "\n", new + "\n"))
        return
    if command == "insert":
        if rel is None:
            raise Drop("insert outside repo")
        line_no = int(args.get("insert_line", -1))
        seen = st.files_seen.get(rel, {})
        if line_no not in seen or not seen[line_no].strip():
            raise Drop("insert at unseen line")
        anchor = seen[line_no]
        new = args.get("new_str", "")
        st.edits += 1
        add(content, "edit_file", {"filepath": rel, "old_string": anchor, "new_string": anchor + "\n" + new.rstrip("\n")},
            obs.edit_file(rel, anchor + "\n", anchor + "\n" + new.rstrip("\n") + "\n"))
        return
    raise Drop(f"unsupported editor command {command}")
