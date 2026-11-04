#!/usr/bin/env python
"""Assemble + render + mask + tokenize SFT records into a training-ready dataset.

    python scripts/build_sft_dataset.py \
        --tokenizer /kaggle/input/gemma-4/.../gemma-4-e4b-it-qat-w4a16-ct \
        --calibration data/calibration.json --profile profiles/e4b_lora.yaml \
        --source data/remapped.jsonl:1.0 --source data/graph_synth.jsonl:2.0 --source data/rft.jsonl:3.0 \
        --max-len 16384 --out data/sft_e4b

Each --source is PATH[:WEIGHT]; WEIGHT is a sampling multiplier (2.0 = every record twice,
0.5 = random half). Records longer than --max-len tokens are dropped (not truncated: a cut
trajectory teaches the model to stop mid-task). Output (np.memmap-friendly):
    input_ids.npy (int32, concatenated), loss_mask.npy (uint8), index.json, stats.json
Ordering implements a length curriculum (short -> long, shuffled within buckets), which
SWE-Lego found better than a learned difficulty scorer.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gemma_swe.calibration import Calibration  # noqa: E402
from gemma_swe.chat_render import RenderMismatch, decode_labels, render_and_mask  # noqa: E402
from gemma_swe.harness_prompt import Budget  # noqa: E402
from gemma_swe.records import assemble, read_jsonl  # noqa: E402
from gemma_swe.submission_builder import TEMPLATE_DIR, load_profile  # noqa: E402
from gemma_swe.tool_schema import required_args, tool_names  # noqa: E402
from gemma_swe.traj_filters import passes  # noqa: E402

_TOK = None
_CTX: dict = {}


def _init(tok_path, ctx):
    global _TOK, _CTX
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(tok_path)
    _CTX = ctx


def _work(rec: dict):
    calib = Calibration(**_CTX["calib"])
    try:
        mode = _CTX["rationale_mode"]
        msgs, tools = assemble(rec, calib, _CTX["instruction"], Budget(**_CTX["budget"]), _CTX["agent_name"],
                               _CTX["description"], rationale_mode=mode)
        r = render_and_mask(_TOK, msgs, tools, enable_thinking=(mode == "thinking"))
    except RenderMismatch as e:
        return {"id": rec.get("id"), "drop": f"render_mismatch: {e}"}
    except Exception as e:  # malformed record
        return {"id": rec.get("id"), "drop": f"error: {type(e).__name__}: {e}"}
    if r.n_tokens > _CTX["max_len"]:
        return {"id": rec.get("id"), "drop": "too_long", "n": r.n_tokens}
    if r.n_label_tokens == 0:
        return {"id": rec.get("id"), "drop": "no_labels"}
    return {"id": rec.get("id"), "source": rec.get("source"), "ids": r.input_ids, "mask": r.labels_mask,
            "meta": rec.get("meta", {})}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, help="dir with tokenizer + chat_template of the SERVED checkpoint")
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--source", action="append", required=True)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--max-calls", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--show", type=int, default=1, help="print labelled segments of N samples")
    ap.add_argument("--rationale-mode", choices=["none", "thinking", "content"], default=None,
                    help="default: 'thinking' if the profile's thinking_budget > 0 else 'none' (see records.assemble)")
    a = ap.parse_args(argv)

    rng = random.Random(a.seed)
    prof = load_profile(a.profile)
    calib = Calibration.load(a.calibration)
    if not calib.is_captured:
        print("WARNING: no captured calibration -> using reconstructed prompts/tools. Run "
              "scripts/capture_harness_prompt.py first for train==serve fidelity.", file=sys.stderr)
    tools = calib.get_tools()
    declared, req = set(tool_names(tools)), required_args(tools)
    instruction = (TEMPLATE_DIR / prof["root"]["instruction"]).read_text(encoding="utf-8")

    recs, dropped = [], Counter()
    for spec in a.source:
        if ":" in spec and not Path(spec).exists():
            path, _, w = spec.rpartition(":")
        else:
            path, w = spec, ""
        w = float(w) if w else 1.0
        rows = list(read_jsonl(path))
        kept = []
        for r in rows:
            ok, why = passes(r, max_calls=a.max_calls, required=req, declared=declared)
            if ok:
                kept.append(r)
            else:
                dropped[f"filter:{why}"] += 1
        reps = int(w)
        out = kept * reps + [r for r in kept if rng.random() < (w - reps)]
        print(f"{path}: {len(rows)} rows, {len(kept)} pass filters, {len(out)} after weight {w}")
        recs += out

    mode = a.rationale_mode or ("thinking" if int(prof["sampling"].get("thinking_budget") or 0) > 0 else "none")
    print(f"rationale mode: {mode}")
    ctx = {"calib": calib.__dict__, "instruction": instruction, "max_len": a.max_len, "rationale_mode": mode,
           "budget": {"time_minutes": float(prof["evaluation"]["max_time_minutes"]),
                      "tool_calls": prof["evaluation"].get("max_tool_calls"),
                      "turns": prof["evaluation"].get("max_turns"),
                      "command_timeout_s": int(prof["evaluation"].get("timeout_seconds", 300))},
           "agent_name": prof["root"]["name"], "description": prof["root"].get("description")}
    with Pool(a.workers, initializer=_init, initargs=(a.tokenizer, ctx)) as pool:
        results = pool.map(_work, recs, chunksize=8)

    good = []
    for r in results:
        if "drop" in r:
            dropped[r["drop"].split(":")[0]] += 1
        else:
            good.append(r)
    if not good:
        sys.exit(f"no usable samples; drops: {dict(dropped)}")

    if a.no_curriculum:
        rng.shuffle(good)
    else:
        good.sort(key=lambda r: len(r["ids"]))
        nb = 8
        buckets = [good[i * len(good) // nb : (i + 1) * len(good) // nb] for i in range(nb)]
        good = []
        for b in buckets:
            rng.shuffle(b)
            good += b

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = sum(len(r["ids"]) for r in good)
    ids = np.empty(total, dtype=np.int32)
    mask = np.empty(total, dtype=np.uint8)
    index, off = [], 0
    for r in good:
        n = len(r["ids"])
        ids[off : off + n] = r["ids"]
        mask[off : off + n] = r["mask"]
        index.append({"offset": off, "length": n, "n_label": int(sum(r["mask"])), "id": r["id"], "source": r["source"],
                      "hop_bucket": (r.get("meta") or {}).get("hop_bucket")})
        off += n
    np.save(out / "input_ids.npy", ids)
    np.save(out / "loss_mask.npy", mask)
    (out / "index.json").write_text(json.dumps(index))
    lens = [x["length"] for x in index]
    stats = {
        "n_samples": len(index), "total_tokens": int(total), "label_tokens": int(mask.sum()),
        "len_p50": int(np.percentile(lens, 50)), "len_p90": int(np.percentile(lens, 90)), "len_max": int(max(lens)),
        "by_source": dict(Counter(x["source"] for x in index)), "dropped": dict(dropped),
        "calibration": "captured" if calib.is_captured else "reconstructed", "max_len": a.max_len, "rationale_mode": mode,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))

    if a.show:
        from transformers import AutoTokenizer

        from gemma_swe.chat_render import Rendered

        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        for r in good[: a.show]:
            segs = decode_labels(tok, Rendered("", r["ids"], r["mask"], []))
            print(f"\n=== labelled segments of {r['id']} ({len(segs)} segments) ===")
            for s in segs[:6]:
                print(repr(s[:400]))


if __name__ == "__main__":
    main()
