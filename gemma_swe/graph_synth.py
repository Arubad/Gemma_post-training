"""Graph-path-grounded trajectory synthesis (teacher-free).

For each training task: gold nodes = innermost defs touched by the gold patch (ast on the real
snapshot); anchors = graph nodes the issue text mentions; path = shortest call-graph path
anchor -> gold. The trajectory walks that path with the harness's own graph tools, then reads,
edits (gold hunks as unique edit_file calls), verifies and submits. Every observation is
produced by actually executing the step on the extracted snapshot (run_command, read_file,
edit_file, git diff) or by emulating the graph tools on the released graph/embeddings, so the
data is consistent with what the agent would see.

Hop depth (anchor -> gold distance) is recorded per trajectory for hop-stratified evaluation,
and hop >= 2 trajectories are oversampled (small models rarely chain expansions; 2610.33958).
"""

from __future__ import annotations

import random
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path

from . import observations as obs
from .constants import is_protected_path
from .graph_utils import (
    CodeGraph,
    Embeddings,
    collect_defs,
    extract_mentions,
    find_anchors,
    gold_nodes_for_file,
    hop_bucket,
    shortest_path,
)
from .harness_prompt import workspace_layout
from .patch_utils import apply_edits, apply_hunks, hunks_to_edits, new_file_content, parse_unified_diff


@dataclass
class SynthConfig:
    max_changed_lines: int = 80
    max_files: int = 3
    max_edits: int = 6
    max_hops: int = 4
    deep_hop_copies: int = 2  # copies for hop >= 2
    p_embed_first: float = 0.5
    cmd_timeout: int = 60


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------- sandbox
class Sandbox:
    """Executes steps on an extracted snapshot, rewriting /workspace <-> real dir."""

    def __init__(self, repo_dir: Path, timeout: int = 60):
        self.dir = repo_dir
        self.timeout = timeout

    def _abs(self, rel: str) -> Path:
        return self.dir / rel.lstrip("/").removeprefix("workspace/")

    def run(self, cmd: str) -> dict:
        real = cmd.replace("/workspace", str(self.dir))
        try:
            p = subprocess.run(["bash", "-c", real], cwd=self.dir, capture_output=True, text=True, timeout=self.timeout)
            out = p.stdout.replace(str(self.dir), "/workspace")
            err = p.stderr.replace(str(self.dir), "/workspace")
            return obs.run_command(out, err, p.returncode)
        except subprocess.TimeoutExpired:
            return obs.run_command("", "", 124, timed_out=True, timeout_s=self.timeout)

    def read(self, rel: str, s: int | None = None, e: int | None = None) -> dict:
        return obs.read_file(rel, self._abs(rel).read_text(encoding="utf-8", errors="replace"), s, e)

    def text(self, rel: str) -> str:
        return self._abs(rel).read_text(encoding="utf-8", errors="replace")

    def edit(self, rel: str, old: str, new: str) -> dict:
        p = self._abs(rel)
        before = p.read_text(encoding="utf-8")
        if before.count(old) != 1:
            raise Skip(f"edit not unique in {rel}")
        after = before.replace(old, new, 1)
        p.write_text(after, encoding="utf-8")
        return obs.edit_file(rel, before, after)

    def write(self, rel: str, content: str) -> dict:
        p = self._abs(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return obs.write_file(rel, content)

    def diff(self) -> str:
        subprocess.run(["git", "add", "-N", "."], cwd=self.dir, capture_output=True)
        p = subprocess.run(["git", "diff", "HEAD"], cwd=self.dir, capture_output=True, text=True)
        return p.stdout


def extract_snapshot(tgz: Path, dest: Path) -> Path:
    with tarfile.open(tgz) as tf:
        try:
            tf.extractall(dest, filter="data")
        except TypeError:
            tf.extractall(dest)
    if (dest / ".git").exists():
        return dest
    subs = [p for p in dest.iterdir() if p.is_dir()]
    if len(subs) == 1 and (subs[0] / ".git").exists():
        return subs[0]
    return dest


def _ensure_git(repo: Path) -> None:
    if not (repo / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=repo)
    subprocess.run(["git", "add", "-A"], cwd=repo, capture_output=True)
    subprocess.run(["git", "-c", "user.email=s@s", "-c", "user.name=s", "commit", "-qm", "baseline", "--allow-empty"],
                   cwd=repo, capture_output=True)


# ----------------------------------------------------------------------------- rationale text
def _short(nid: str, keep: int = 2) -> str:
    parts = nid.split(".")
    return ".".join(parts[-keep:]) if len(parts) >= keep else nid


def _title(problem: str) -> str:
    t = problem.strip().splitlines()[0] if problem.strip() else "the issue"
    t = re.sub(r"^[^\w`]+", "", t)  # drop emoji / bullet prefixes
    return t[:120]


R_LOCATE_EMB = [
    "The issue is about `{a}`; I'll look it up in the code graph to find the related code.",
    "Start from `{a}`, which the issue mentions, and find similar code via the embeddings.",
]
R_LOCATE_GREP = [
    "The issue mentions `{a}`; I'll find where it is defined.",
    "First locate the definition of `{a}`.",
]
R_LOCATE_KW = [
    "I'll search the source for `{a}` to find the relevant module.",
]
R_HOP_FIRST = [
    "See which functions `{cur}` calls and is called by.",
    "Follow the call graph from `{cur}` to find the code that needs changing.",
]
R_HOP_NEXT = [
    "`{prev}` {rel} `{cur}`; follow the call chain from `{cur}`.",
    "The behaviour goes through `{cur}` ({prev} {rel} it); check its neighbours next.",
]
R_READ = [
    "`{g}` looks like where the change belongs; I'll read it in `{f}`.",
    "Read `{g}` in `{f}` before editing.",
]
R_READ_AFTER_HOP = [
    "`{prev}` {rel} `{g}`, which looks like where the change belongs; reading it in `{f}`.",
]
R_EDIT = [
    "Make the minimal change in `{f}`.",
    "Apply the fix in `{f}`, keeping the surrounding code unchanged.",
    "Update `{f}` accordingly.",
]
R_VERIFY = ["Check the edited files still compile.", "Quick syntax check of the changed files."]
R_SUBMIT = ["The change is in place and the files compile; submitting the patch."]


def _pick(rng: random.Random, opts: list[str], **kw) -> str:
    return rng.choice(opts).format(**kw)


# ----------------------------------------------------------------------------- main
def synth_task(task: dict, snapshots_dir: Path, graphs_dir: Path, emb_dir: Path | None,
               cfg: SynthConfig, seed: int = 0, workdir: Path | None = None) -> list[dict]:
    iid = task["instance_id"]
    tgz = snapshots_dir / f"{iid}.tgz"
    gpath = graphs_dir / f"{iid}.json"
    if not tgz.exists() or not gpath.exists():
        raise Skip("missing snapshot or graph")
    fps = [fp for fp in parse_unified_diff(task["patch"]) if not is_protected_path(fp.path)]
    if not fps:
        raise Skip("no source changes")
    if len(fps) > cfg.max_files or sum(fp.n_changed() for fp in fps) > cfg.max_changed_lines:
        raise Skip("patch too large")
    if any(fp.is_deleted for fp in fps):
        raise Skip("file deletion")
    graph = CodeGraph.load(gpath)
    emb = None
    if emb_dir is not None and (emb_dir / f"{iid}.npz").exists():
        try:
            emb = Embeddings.load(emb_dir / f"{iid}.npz")
        except Exception:
            emb = None

    tmp_root = Path(tempfile.mkdtemp(prefix=f"synth_{iid}_", dir=workdir))
    try:
        repo = extract_snapshot(tgz, tmp_root / "snap")
        _ensure_git(repo)
        originals = {fp.path: (repo / fp.path).read_text(encoding="utf-8") for fp in fps if not fp.is_new and (repo / fp.path).exists()}
        golds: list[str] = []
        for fp in fps:
            if fp.path in originals:
                lines = sorted({ln for h in fp.hunks for ln in h.changed_old_linenos()})
                golds += [g for g in gold_nodes_for_file(graph, fp.path, originals[fp.path], lines) if g not in golds]
        mentions = extract_mentions(task["problem_statement"])
        anchors = find_anchors(graph, mentions)
        adj = graph.undirected_adj()
        path = shortest_path(adj, anchors, set(golds), max_depth=cfg.max_hops) if golds else None
        hops = (len(path) - 1) if path else None
        layout = workspace_layout(repo)

        # edit plan (validated against positional application of the gold hunks)
        plans: list[tuple[str, str, object]] = []  # (kind, path, payload)
        for fp in fps:
            if fp.is_new:
                content = new_file_content(fp)
                if content.count("\n") > 300:
                    raise Skip("new file too long")
                plans.append(("write", fp.path, content))
                continue
            if fp.path not in originals:
                raise Skip(f"patched file missing in snapshot: {fp.path}")
            orig = originals[fp.path]
            want = apply_hunks(orig, fp.hunks)
            edits = None
            for gap in (3, 10**6):
                try:
                    e = hunks_to_edits(orig, fp.hunks, merge_gap=gap)
                    if apply_edits(orig, e) == want:
                        edits = e
                        break
                except ValueError:
                    continue
            if edits is None:
                raise Skip(f"could not build unique edits for {fp.path}")
            for old, new in edits:
                plans.append(("edit", fp.path, (old, new)))
        if sum(1 for p in plans if p[0] in ("edit", "write")) > cfg.max_edits:
            raise Skip("too many edits")

        copies = cfg.deep_hop_copies if (hops is not None and hops >= 2) else 1
        records = []
        for c in range(copies):
            rng = random.Random(f"{iid}:{seed}:{c}")
            # fresh working tree for each copy
            subprocess.run(["git", "checkout", "-q", "--", "."], cwd=repo, capture_output=True)
            subprocess.run(["git", "clean", "-qfd"], cwd=repo, capture_output=True)
            sb = Sandbox(repo, cfg.cmd_timeout)
            turns = _render(task, sb, graph, emb, golds, anchors, mentions, path, plans, originals, rng, cfg)
            records.append({
                "id": f"graph:{iid}:{c}",
                "source": "graph_synth",
                "repo": task["repo"],
                "problem_statement": task["problem_statement"],
                "hints": task.get("hints_text") or None,
                "layout": layout,
                "turns": turns,
                "meta": {"instance_id": iid, "hops": hops, "hop_bucket": hop_bucket(hops), "anchors": anchors[:4],
                         "golds": golds, "path": path, "copy": c},
            })
        return records
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


def _render(task, sb: Sandbox, graph: CodeGraph, emb, golds, anchors, mentions, path, plans, originals,
            rng: random.Random, cfg: SynthConfig) -> list[dict]:
    turns: list[dict] = []

    def step(content: str, name: str, args: dict, result: dict):
        turns.append({"role": "assistant", "content": content, "tool_calls": [{"name": name, "arguments": args}], "train": True})
        turns.append({"role": "tool", "name": name, "result": result})
        return result

    files = list(dict.fromkeys(p[1] for p in plans))
    gold_file = files[0]

    # ---------------- LOCATE
    if path:
        a = path[0]
        a_short = _short(a)
        if emb is not None and rng.random() < cfg.p_embed_first:
            res = emb.search(graph, a_short, k=5)
            payload = [(n, graph.nodes.get(n, {}).get("text", ""), s) for n, s in res]
            step(_pick(rng, R_LOCATE_EMB, a=a_short), "search_similar_code", {"query": a_short, "k": 5}, obs.search_similar(a_short, payload))
        else:
            nm = a.split(".")[-1]
            step(_pick(rng, R_LOCATE_GREP, a=nm), "run_command",
                 {"command": f'grep -rn "def {nm}\\b\\|class {nm}\\b" --include=*.py . | head -20'},
                 sb.run(f'grep -rn "def {nm}\\b\\|class {nm}\\b" --include=*.py . | head -20'))
        # ---------------- TRAVERSE
        prev_rel = None
        for i, (cur, nxt) in enumerate(zip(path[:-1], path[1:])):
            shown = graph.neighbors(cur, max_neighbors=50)
            if nxt not in shown:
                raise Skip("next hop truncated from neighbour list")
            if prev_rel is None:
                why = _pick(rng, R_HOP_FIRST, cur=_short(cur))
            else:
                why = _pick(rng, R_HOP_NEXT, prev=_short(path[i - 1]), rel=prev_rel, cur=_short(cur))
            step(why, "get_code_neighbors", {"node": _short(cur)}, obs.neighbors(graph.resolve(_short(cur)) or cur, shown))
            prev_rel = "calls" if any(e.dst == nxt for e in graph.out.get(cur, [])) else "is called by"
        hop_ctx = (path[-2], prev_rel) if len(path) > 1 else None
    else:
        kws = [m for m in mentions if not m[0].isupper() or "_" in m] or mentions
        hit = None
        for kw in kws[:3]:
            r = sb.run(f'grep -rn "{kw.split(".")[-1]}" --include=*.py . | grep -v "^./tests/" | head -20')
            if gold_file in (r.get("stdout") or ""):
                hit = (kw.split(".")[-1], r)
                break
        if hit is None:
            raise Skip("no anchor path and keyword grep does not surface the gold file")
        cmd = f'grep -rn "{hit[0]}" --include=*.py . | grep -v "^./tests/" | head -20'
        step(_pick(rng, R_LOCATE_KW, a=hit[0]), "run_command", {"command": cmd}, hit[1])
        hop_ctx = None

    # ---------------- READ
    for f in files:
        if f not in originals:
            continue
        text = originals[f]
        n = text.count("\n") + 1
        defs = collect_defs(text)
        g = next((x for x in golds if x.split(".")[-1] in {d.qualname.split(".")[-1] for d in defs}), None)
        d = next((dd for dd in defs if g.endswith(dd.qualname)), None) if g else None
        # every line an edit touches must have been read first (never edit unseen code)
        spans = []
        for kind, pth, payload in plans:
            if pth == f and kind == "edit":
                old = payload[0]
                a0 = text[: text.find(old)].count("\n") + 1
                spans.append((max(1, a0 - 3), min(n, a0 + old.count("\n") + 3)))
        if d:
            spans.append((max(1, d.start - 3), min(n, d.end + 3)))
        if not spans:
            spans.append((1, min(n, 60)))
        spans.sort()
        merged = [list(spans[0])]
        for a0, b0 in spans[1:]:
            if a0 <= merged[-1][1] + 10:
                merged[-1][1] = max(merged[-1][1], b0)
            else:
                merged.append([a0, b0])
        windows = []
        for a0, b0 in merged:  # split windows at the 150-line read cap
            while a0 <= b0:
                windows.append((a0, min(b0, a0 + 149)))
                a0 += 150
        if len(windows) > 4:
            raise Skip("edited regions too spread out to read within budget")
        label = _short(g) if g else Path(f).name
        for wi, (ws, we) in enumerate(windows):
            if wi == 0 and hop_ctx and g == path[-1]:
                why = _pick(rng, R_READ_AFTER_HOP, prev=_short(hop_ctx[0]), rel=hop_ctx[1], g=label, f=f)
                hop_ctx = None
            elif wi == 0:
                why = _pick(rng, R_READ, g=label, f=f)
            else:
                why = f"Read the other part of `{f}` that needs changing."
            res = sb.read(f, ws, we)
            if res.get("is_truncated"):
                raise Skip("read window exceeds the 10k-char read cap")
            step(why, "read_file", {"filepath": f, "start_line": ws, "end_line": we}, res)

    # ---------------- EDIT
    for kind, pth, payload in plans:
        if kind == "write":
            step(f"Create `{pth}`, which the change needs.", "write_file", {"filepath": pth, "content": payload}, sb.write(pth, payload))
        else:
            old, new = payload
            step(_pick(rng, R_EDIT, f=pth), "edit_file", {"filepath": pth, "old_string": old, "new_string": new}, sb.edit(pth, old, new))

    # ---------------- VERIFY
    py_files = [f for f in files if f.endswith(".py")]
    if py_files:
        cmd = "python -m py_compile " + " ".join(py_files) + " && echo OK"
        step(rng.choice(R_VERIFY), "run_command", {"command": cmd}, sb.run(cmd))
    if rng.random() < 0.5:
        step("Review the full diff before submitting.", "run_command", {"command": "git diff"}, sb.run("git diff"))

    # ---------------- SUBMIT
    patch = sb.diff()
    step(rng.choice(R_SUBMIT), "submit_patch", {}, obs.submit_patch(patch))
    targets = ", ".join(f"`{_short(g)}`" for g in golds[:3]) or ", ".join(f"`{f}`" for f in files)
    turns.append({"role": "assistant", "content": f"Updated {targets} to address: {_title(task['problem_statement'])}", "train": True})
    return turns
