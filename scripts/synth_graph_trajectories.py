#!/usr/bin/env python
"""Synthesize graph-path-grounded trajectories from the public training tasks (CPU only).

    python scripts/synth_graph_trajectories.py --tasks data/tasks_train.jsonl \
        --comp-dir /kaggle/input/gemma-4-developer-agent --out data/graph_synth.jsonl

Only TRAIN-split tasks should be used (default excludes HELDOUT_REPOS) so local held-out
evaluation stays clean. Writes <out> plus <out>.stats.json (per-hop-bucket counts, skip reasons).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.constants import HELDOUT_REPOS, comp_data_dir, tasks_path  # noqa: E402
from gemma_swe.graph_synth import Skip, SynthConfig, synth_task  # noqa: E402
from gemma_swe.records import read_jsonl  # noqa: E402

_ARGS = None


def _init(args):
    global _ARGS
    _ARGS = args


def _one(task):
    a = _ARGS
    comp = Path(a["comp_dir"])
    try:
        recs = synth_task(task, comp / "snapshots", comp / "graphs", comp / "embeddings",
                          SynthConfig(max_changed_lines=a["max_changed"], deep_hop_copies=a["deep_copies"]),
                          seed=a["seed"], workdir=Path(a["workdir"]) if a["workdir"] else None)
        return task["instance_id"], recs, None
    except Skip as e:
        return task["instance_id"], [], str(e)
    except Exception as e:
        return task["instance_id"], [], f"error: {type(e).__name__}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default=str(tasks_path()))
    ap.add_argument("--comp-dir", default=str(comp_data_dir() or ""))
    ap.add_argument("--out", required=True)
    ap.add_argument("--include-heldout", action="store_true")
    ap.add_argument("--max-changed", type=int, default=80)
    ap.add_argument("--deep-copies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--workdir", default="", help="scratch dir for snapshot extraction (default: system tmp)")
    a = ap.parse_args(argv)
    if not a.comp_dir or not Path(a.comp_dir, "snapshots").exists():
        sys.exit(f"competition data dir with snapshots/ graphs/ embeddings/ not found: {a.comp_dir!r}")

    tasks = [t for t in read_jsonl(a.tasks) if a.include_heldout or t["repo"] not in HELDOUT_REPOS]
    args = {"comp_dir": a.comp_dir, "max_changed": a.max_changed, "deep_copies": a.deep_copies,
            "seed": a.seed, "workdir": a.workdir}
    skips, buckets, n = Counter(), Counter(), 0
    with Pool(a.workers, initializer=_init, initargs=(args,)) as pool, open(a.out, "w", encoding="utf-8") as fo:
        for iid, recs, why in pool.imap_unordered(_one, tasks):
            if why:
                skips[why.split(":")[0] if why.startswith("error") else why] += 1
                continue
            for r in recs:
                fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                buckets[r["meta"]["hop_bucket"]] += 1
                n += 1
    stats = {"tasks": len(tasks), "records": n, "by_hop_bucket": dict(buckets), "skips": dict(skips.most_common())}
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
