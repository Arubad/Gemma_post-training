#!/usr/bin/env python
"""Per-task hop depth between issue-mentioned graph nodes and gold-patch nodes (CPU).

    python scripts/hop_map.py --comp-dir /kaggle/input/gemma-4-developer-agent --out data/hop_map.json

Used by eval_report.py to stratify resolve rates by localization difficulty (hop 0/1/2/3+/unreachable).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.constants import comp_data_dir, is_protected_path, tasks_path  # noqa: E402
from gemma_swe.graph_synth import extract_snapshot  # noqa: E402
from gemma_swe.graph_utils import CodeGraph, extract_mentions, find_anchors, gold_nodes_for_file, hop_bucket, shortest_path  # noqa: E402
from gemma_swe.patch_utils import parse_unified_diff  # noqa: E402
from gemma_swe.records import read_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(tasks_path()))
    ap.add_argument("--comp-dir", default=str(comp_data_dir() or ""))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    comp = Path(a.comp_dir)
    out, buckets = {}, Counter()
    for t in read_jsonl(a.tasks):
        iid = t["instance_id"]
        try:
            g = CodeGraph.load(comp / "graphs" / f"{iid}.json")
        except Exception as e:
            out[iid] = {"error": str(e), "hop_bucket": "no_graph"}
            buckets["no_graph"] += 1
            continue
        tmp = Path(tempfile.mkdtemp())
        try:
            repo = extract_snapshot(comp / "snapshots" / f"{iid}.tgz", tmp)
            golds = []
            for fp in parse_unified_diff(t["patch"]):
                if fp.is_new or is_protected_path(fp.path) or not (repo / fp.path).exists():
                    continue
                lines = sorted({ln for h in fp.hunks for ln in h.changed_old_linenos()})
                golds += gold_nodes_for_file(g, fp.path, (repo / fp.path).read_text(encoding="utf-8", errors="replace"), lines)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        anchors = find_anchors(g, extract_mentions(t["problem_statement"]))
        path = shortest_path(g.undirected_adj(), anchors, set(golds)) if golds else None
        hops = len(path) - 1 if path else None
        b = hop_bucket(hops) if golds else "no_gold_node"
        buckets[b] += 1
        out[iid] = {"repo": t["repo"], "hops": hops, "hop_bucket": b, "golds": golds, "anchors": anchors[:5], "path": path}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(dict(buckets), indent=2))


if __name__ == "__main__":
    main()
