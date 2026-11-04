#!/usr/bin/env python
"""Download public SWE-agent trajectories from Hugging Face and re-map them to the harness schema.

Run in an INTERNET-ENABLED CPU notebook (or locally), then upload the output as a Kaggle Dataset.

    python scripts/prepare_hf_trajectories.py \
        --dataset nebius/SWE-rebench-openhands-trajectories:train:6000 \
        --dataset SWE-bench/SWE-smith-trajectories:tool:3000 \
        --out data/remapped.jsonl --per-instance 2

--dataset NAME[:SPLIT[:LIMIT]]  (LIMIT = max raw rows scanned, streamed)
--inspect prints the first raw record's structure and exits (use it when a schema changes).
Licences: nebius / nvidia sets are CC-BY-4.0, SWE-smith MIT (attribute them in the writeup).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.constants import HELDOUT_REPOS  # noqa: E402
from gemma_swe.remap_openhands import Drop, RemapConfig, remap  # noqa: E402
from gemma_swe.tool_schema import default_tools, required_args  # noqa: E402
from gemma_swe.traj_filters import passes  # noqa: E402


def _shape(x, depth=0):
    if depth > 3:
        return "..."
    if isinstance(x, dict):
        return {k: _shape(v, depth + 1) for k, v in list(x.items())[:12]}
    if isinstance(x, list):
        return [_shape(x[0], depth + 1), f"... len={len(x)}"] if x else []
    if isinstance(x, str):
        return f"str[{len(x)}]: {x[:60]!r}"
    return type(x).__name__


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--per-instance", type=int, default=2, help="max trajectories per instance_id")
    ap.add_argument("--exclude-repos", default=",".join(sorted(HELDOUT_REPOS)),
                    help="comma list; held-out eval repos must not leak into training")
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args(argv)

    from datasets import load_dataset

    exclude = {r.strip().lower() for r in a.exclude_repos.split(",") if r.strip()}
    req = required_args(default_tools())
    cfg = RemapConfig(max_tool_calls=a.max_calls)
    drops: Counter = Counter()
    per_inst: dict[str, int] = defaultdict(int)
    n_out = 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for spec in a.dataset:
            parts = spec.split(":")
            name, split = parts[0], (parts[1] if len(parts) > 1 and parts[1] else "train")
            limit = int(parts[2]) if len(parts) > 2 and parts[2] else None
            ds = load_dataset(name, split=split, streaming=True)
            kept = 0
            for i, raw in enumerate(ds):
                if limit and i >= limit:
                    break
                if a.inspect:
                    print(json.dumps(_shape(raw), indent=2))
                    return
                raw = dict(raw)
                raw["_source"] = name.split("/")[-1]
                try:
                    rec = remap(raw, cfg)
                except Drop as e:
                    drops[f"{name}:{str(e).split(':')[0]}"] += 1
                    continue
                except Exception as e:  # unexpected schema
                    drops[f"{name}:exception:{type(e).__name__}"] += 1
                    continue
                if rec["repo"].lower() in exclude:
                    drops[f"{name}:excluded_repo"] += 1
                    continue
                ok, why = passes(rec, max_calls=a.max_calls, required=req)
                if not ok:
                    drops[f"{name}:filter:{why}"] += 1
                    continue
                key = rec["meta"].get("instance_id") or rec["id"]
                if per_inst[key] >= a.per_instance:
                    drops[f"{name}:per_instance_cap"] += 1
                    continue
                per_inst[key] += 1
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                n_out += 1
            print(f"{name}[{split}]: kept {kept}")
    stats = {"written": n_out, "drops": dict(drops.most_common())}
    Path(a.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
