import difflib
import io
import json
import subprocess
import tarfile
from pathlib import Path

import numpy as np

from gemma_swe.constants import REPO_ROOT, is_protected_path
from gemma_swe.graph_synth import SynthConfig, synth_task
from gemma_swe.graph_utils import CodeGraph, Embeddings, collect_defs, extract_mentions, find_anchors, gold_nodes_for_file, shortest_path
from gemma_swe.patch_utils import apply_edits, apply_hunks, hunks_to_edits, parse_unified_diff

ORIG = """import os


def helper(x):
    return x + 1


def parse_header(value):
    parts = value.split(";")
    return parts[0]


class Session:
    def request(self, url):
        h = parse_header(url)
        return helper(len(h))

    def close(self):
        return None
"""
NEW = ORIG.replace('    return parts[0]\n', '    if not parts:\n        return ""\n    return parts[0].strip()\n').replace(
    "    def close(self):\n        return None\n", "    def close(self):\n        self.closed = True\n        return None\n")


def mkpatch(path, a, b):
    return "".join(difflib.unified_diff(a.splitlines(True), b.splitlines(True), f"a/{path}", f"b/{path}", n=3))


def test_protected_paths():
    assert is_protected_path("tests/test_x.py") and is_protected_path("pkg/test_y.py") and is_protected_path("conftest.py")
    assert is_protected_path("pyproject.toml") and is_protected_path("a/b/tests/helpers.py")
    assert not is_protected_path("pkg/models.py") and not is_protected_path(".github/workflows/x.py")


def test_hunks_to_edits_roundtrip():
    fp = parse_unified_diff(mkpatch("pkg/http.py", ORIG, NEW))[0]
    assert apply_hunks(ORIG, fp.hunks) == NEW
    edits = hunks_to_edits(ORIG, fp.hunks)
    assert apply_edits(ORIG, edits) == NEW
    for old, _ in edits:
        assert ORIG.count(old) == 1


def test_pure_insertion_and_new_file():
    new = ORIG.replace("import os\n", "import os\nimport re\n")
    fp = parse_unified_diff(mkpatch("m.py", ORIG, new))[0]
    assert apply_edits(ORIG, hunks_to_edits(ORIG, fp.hunks)) == new
    nf = parse_unified_diff("--- a/new.py\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n")[0]
    assert nf.is_new and apply_hunks("", nf.hunks) == "x = 1\ny = 2\n"


def test_all_public_gold_patches_parse():
    rows = [json.loads(l) for l in open(REPO_ROOT / "tasks.jsonl")]
    for r in rows:
        for fp in parse_unified_diff(r["patch"]):
            for h in fp.hunks:
                assert len(h.old_lines) == h.old_len and len(h.new_lines) == h.new_len, r["instance_id"]


# ------------------------------------------------------------------------------- graph
def graph_json():
    nodes = [
        {"id": "pkg.http", "name": "pkg.http", "text": ORIG},
        {"id": "pkg.http.helper", "name": "pkg.http.helper", "text": "def helper(x):\n    return x + 1\n"},
        {"id": "pkg.http.parse_header", "name": "pkg.http.parse_header", "text": "def parse_header(value):\n    parts = value.split(';')\n"},
        {"id": "pkg.http.Session", "name": "pkg.http.Session", "text": "class Session: ..."},
        {"id": "pkg.http.Session.request", "name": "pkg.http.Session.request", "text": "def request(self, url): ..."},
        {"id": "pkg.http.Session.close", "name": "pkg.http.Session.close", "text": "def close(self): ..."},
        {"id": "pkg.api.get", "name": "pkg.api.get", "text": "def get(url):\n    return Session().request(url)\n"},
    ]
    edges = [
        {"source": "pkg.http.Session.request", "target": "pkg.http.parse_header", "type": "calls", "key": 0},
        {"source": "pkg.http.Session.request", "target": "pkg.http.helper", "type": "calls", "key": 0},
        {"source": "pkg.api.get", "target": "pkg.http.Session.request", "type": "calls", "key": 0},
    ]
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes, "edges": edges}


def test_graph_resolution_and_paths(tmp_path):
    p = tmp_path / "g.json"
    p.write_text(json.dumps(graph_json()))
    g = CodeGraph.load(p)
    assert g.resolve("Session.request") == "pkg.http.Session.request"
    assert g.resolve("parse_header") == "pkg.http.parse_header"
    assert set(g.neighbors("Session.request")) == {"pkg.api.get", "pkg.http.parse_header", "pkg.http.helper"}
    path = shortest_path(g.undirected_adj(), ["pkg.api.get"], {"pkg.http.parse_header"})
    assert path == ["pkg.api.get", "pkg.http.Session.request", "pkg.http.parse_header"]
    defs = {d.qualname for d in collect_defs(ORIG)}
    assert {"helper", "parse_header", "Session", "Session.request", "Session.close"} <= defs
    fp = parse_unified_diff(mkpatch("pkg/http.py", ORIG, NEW))[0]
    lines = sorted({ln for h in fp.hunks for ln in h.changed_old_linenos()})
    assert set(gold_nodes_for_file(g, "pkg/http.py", ORIG, lines)) == {"pkg.http.parse_header", "pkg.http.Session.close"}
    m = extract_mentions("Calling `pkg.api.get` with an empty header crashes in api.get()")
    assert "pkg.api.get" in m
    assert find_anchors(g, m)[0] == "pkg.api.get"


def test_embeddings_search(tmp_path):
    ids = ["a.x", "a.y", "b.z"]
    np.savez(tmp_path / "e.npz", **{"a.x": np.array([1, 0], np.float32), "a.y": np.array([0.9, 0.1], np.float32),
                                    "b.z": np.array([0, 1], np.float32)})
    e = Embeddings.load(tmp_path / "e.npz")
    res = e.search(None, "a.x", k=2)
    assert res[0][0] == "a.y" and all(n != "a.x" for n, _ in res)


def test_synth_end_to_end(tmp_path):
    """Fabricate a snapshot + graph + embeddings + task and synthesize a graph-path trajectory."""
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "__init__.py").write_text("")
    (repo / "pkg" / "http.py").write_text(ORIG)
    (repo / "pkg" / "api.py").write_text("from .http import Session\n\n\ndef get(url):\n    return Session().request(url)\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=a", "commit", "-qm", "base"], cwd=repo, check=True)
    comp = tmp_path / "comp"
    for d in ("snapshots", "graphs", "embeddings"):
        (comp / d).mkdir(parents=True)
    with tarfile.open(comp / "snapshots" / "pkg_1.tgz", "w:gz") as tf:
        tf.add(repo, arcname=".")
    (comp / "graphs" / "pkg_1.json").write_text(json.dumps(graph_json()))
    ids = [n["id"] for n in graph_json()["nodes"]]
    rng = np.random.default_rng(0)
    np.savez(comp / "embeddings" / "pkg_1.npz", **{i: rng.normal(size=8).astype(np.float32) for i in ids})
    task = {"instance_id": "pkg_1", "repo": "acme/pkg", "problem_statement": "Calling `pkg.api.get` with an empty header crashes.",
            "patch": mkpatch("pkg/http.py", ORIG, NEW), "hints_text": ""}
    recs = synth_task(task, comp / "snapshots", comp / "graphs", comp / "embeddings", SynthConfig(), seed=0)
    assert len(recs) == 2  # hop 2 -> deep copies
    r = recs[0]
    assert r["meta"]["hops"] == 2 and r["meta"]["hop_bucket"] == "2"
    names = [tc["name"] for t in r["turns"] if t["role"] == "assistant" for tc in t.get("tool_calls", [])]
    assert names.count("get_code_neighbors") == 2 and "read_file" in names and names[-1] == "submit_patch"
    assert names.count("edit_file") >= 1
    sub = [t for t in r["turns"] if t["role"] == "tool" and t["name"] == "submit_patch"][0]["result"]
    assert sub["files_changed"] == 1 and sub["patch_size"] > 0
    comp_res = [t for t in r["turns"] if t["role"] == "tool" and t["name"] == "run_command"]
    assert any("OK" in (t["result"].get("stdout") or "") for t in comp_res)
    assert r["turns"][-1]["role"] == "assistant" and "tool_calls" not in r["turns"][-1]
