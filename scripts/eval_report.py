#!/usr/bin/env python
"""Compare local_eval.py runs: resolve rate mean +- std over seeds, per repo and per hop bucket,
plus behaviour metrics from the proxy logs (submission rate, loops, calls per task).

    python scripts/eval_report.py --run e4b_prompt=/kaggle/working/eval/e4b_prompt \
        --run e4b_lora=/kaggle/working/eval/e4b_lora --hop-map data/hop_map.json --tasks data/tasks_heldout.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.harvest import chains, conversations, instance_for, is_resolved, load_results  # noqa: E402
from gemma_swe.records import read_jsonl  # noqa: E402
from gemma_swe.traj_filters import max_identical_run  # noqa: E402


def behaviour(log: Path, tasks: list[dict]) -> dict:
    sub, loops, calls, n = 0, 0, [], 0
    for _, recs in conversations(log).items():
        if instance_for(recs[0]["request"]["messages"], tasks) is None:
            continue  # sub-agent conversation
        n += 1
        msgs = chains(recs)[-1]["messages"]
        seq = [(tc["function"]["name"], tc["function"].get("arguments", "")) for m in msgs if m.get("role") == "assistant"
               for tc in m.get("tool_calls") or []]
        sub += any(s[0] == "submit_patch" for s in seq)
        loops += max_identical_run(seq) >= 3
        calls.append(sum(1 for s in seq if s[0] not in ("submit_patch", "get_status")))
    return {"submit_rate": sub / n if n else None, "loop_rate": loops / n if n else None,
            "mean_calls": statistics.mean(calls) if calls else None}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="NAME=EVAL_DIR")
    ap.add_argument("--hop-map", default=None)
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args(argv)
    tasks = list(read_jsonl(a.tasks))
    repo_of = {t["instance_id"]: t["repo"] for t in tasks}
    hop = {k: v.get("hop_bucket") for k, v in json.loads(Path(a.hop_map).read_text()).items()} if a.hop_map else {}
    report = {}
    for spec in a.run:
        name, d = spec.split("=", 1)
        d = Path(d)
        per_seed, by_repo, by_hop, beh = [], defaultdict(list), defaultdict(list), []
        for sd in sorted(d.glob("seed_*")):
            if not sd.is_dir():
                continue
            res = load_results(sd)
            if not res:
                continue
            per_seed.append(sum(is_resolved(r) for r in res.values()) / len(res))
            for iid, r in res.items():
                by_repo[repo_of.get(iid, "?")].append(is_resolved(r))
                by_hop[hop.get(iid, "n/a")].append(is_resolved(r))
            log = d / f"proxy_{sd.name}.jsonl"
            if log.exists():
                beh.append(behaviour(log, tasks))
        mean = lambda xs: round(statistics.mean(xs), 4) if xs else None  # noqa: E731
        report[name] = {
            "seeds": len(per_seed), "rate_mean": mean(per_seed),
            "rate_std": round(statistics.pstdev(per_seed), 4) if len(per_seed) > 1 else 0.0,
            "by_repo": {k: mean([float(x) for x in v]) for k, v in sorted(by_repo.items())},
            "by_hop_bucket": {k: {"rate": mean([float(x) for x in v]), "n": len(v)} for k, v in sorted(by_hop.items())},
            "behaviour": {k: mean([b[k] for b in beh if b[k] is not None]) for k in ("submit_rate", "loop_rate", "mean_calls")} if beh else {},
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
