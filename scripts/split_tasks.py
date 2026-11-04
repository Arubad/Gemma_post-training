#!/usr/bin/env python
"""Split the 129 public tasks into train / held-out (whole unseen repo) / smoke subsets.

    python scripts/split_tasks.py --out-dir data
-> data/tasks_train.jsonl  (fastapi + requests + httpx: SFT synthesis + RFT rollouts)
   data/tasks_heldout.jsonl (Textualize/rich: never trained on; proxy for private hidden repos)
   data/tasks_smoke.jsonl   (6 small held-out tasks for quick end-to-end checks)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.constants import HELDOUT_REPOS, tasks_path  # noqa: E402
from gemma_swe.patch_utils import parse_unified_diff  # noqa: E402
from gemma_swe.records import read_jsonl, write_jsonl  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(tasks_path()))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--heldout-repos", default=",".join(sorted(HELDOUT_REPOS)))
    ap.add_argument("--n-smoke", type=int, default=6)
    a = ap.parse_args(argv)
    held = {r.strip() for r in a.heldout_repos.split(",") if r.strip()}
    tasks = list(read_jsonl(a.tasks))
    train = [t for t in tasks if t["repo"] not in held]
    heldout = [t for t in tasks if t["repo"] in held]
    size = lambda t: sum(fp.n_changed() for fp in parse_unified_diff(t["patch"]))  # noqa: E731
    smoke = sorted(heldout, key=size)[: a.n_smoke]
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("heldout", heldout), ("smoke", smoke)):
        write_jsonl(out / f"tasks_{name}.jsonl", rows)
    print(json.dumps({"train": len(train), "heldout": len(heldout), "smoke": [t["instance_id"] for t in smoke]}, indent=2))


if __name__ == "__main__":
    main()
