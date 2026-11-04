#!/usr/bin/env python
"""Harvest resolved rollouts from local_eval.py runs into exact-format SFT records (RFT round).

    python scripts/harvest_rft.py --eval-dir /kaggle/working/eval/rft_round1 \
        --tasks data/tasks_train.jsonl --out data/rft_round1.jsonl --per-task 2

Pairs <eval-dir>/proxy_seed_<k>.jsonl with <eval-dir>/seed_<k>/task_results.jsonl. Keeps only
resolved, loop-free, submitted trajectories; at most --per-task per instance (shortest first) so
easy tasks do not dominate. Held-out tasks are refused unless --allow-heldout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.constants import HELDOUT_REPOS  # noqa: E402
from gemma_swe.harvest import harvest  # noqa: E402
from gemma_swe.records import read_jsonl, write_jsonl  # noqa: E402
from gemma_swe.traj_filters import passes  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", action="append", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--max-calls", type=int, default=40)
    ap.add_argument("--allow-heldout", action="store_true")
    a = ap.parse_args(argv)
    tasks = list(read_jsonl(a.tasks))
    repo_of = {t["instance_id"]: t["repo"] for t in tasks}
    pool, why = defaultdict(list), Counter()
    for ed in a.eval_dir:
        ed = Path(ed)
        for log in sorted(ed.glob("proxy_seed_*.jsonl")):
            seed = log.stem.split("_")[-1]
            rdir = ed / f"seed_{seed}"
            for rec in harvest(log, rdir, tasks, f"{ed.name}/s{seed}"):
                iid = rec["meta"]["instance_id"]
                if repo_of.get(iid) in HELDOUT_REPOS and not a.allow_heldout:
                    why["heldout_refused"] += 1
                    continue
                if rec["meta"]["n_chains"] > 1 and rec["meta"]["chain"] < rec["meta"]["n_chains"] - 1:
                    why["pre_compaction_chain_kept"] += 1
                ok, w = passes(rec, max_calls=a.max_calls)
                if not ok:
                    why[f"filter:{w}"] += 1
                    continue
                pool[iid].append(rec)
    out = []
    for iid, recs in pool.items():
        recs.sort(key=lambda r: sum(1 for m in r["messages"] if m.get("role") == "assistant"))
        seen, kept = set(), 0
        for r in recs:
            sig = json.dumps([m.get("tool_calls") and [tc["function"]["name"] for tc in m["tool_calls"]]
                              for m in r["messages"] if m.get("role") == "assistant"])
            if sig in seen:
                continue
            seen.add(sig)
            out.append(r)
            kept += 1
            if kept >= a.per_task:
                break
    n = write_jsonl(a.out, out)
    stats = {"records": n, "tasks_with_pass": len(pool), "notes": dict(why)}
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
