#!/usr/bin/env python
"""Local evaluation (and RFT rollout generation) on Kaggle L4x4, scorer-faithful.

Starts vLLM with the scorer's flags + the submission's adapters on :8001, puts the logging proxy
on :8000 (where the harness looks for its model server), and runs `swegemma eval` once per seed.

    python scripts/local_eval.py --submission-dir /kaggle/working/sub_e4b \
        --model-dir /kaggle/input/gemma-4/transformers/gemma-4-e4b-it-qat-w4a16-ct/1 \
        --tasks data/tasks_heldout.jsonl --snapshots-dir $COMP/snapshots \
        --out /kaggle/working/eval/e4b_lora --seeds 0 1 2 --concurrency 4

Outputs per seed: <out>/seed_<k>/{summary.json, task_results.jsonl, ...} and <out>/proxy_seed_<k>.jsonl
(the proxy log doubles as RFT data; see harvest_rft.py). Prints mean +- std resolve rate.

For RFT rollouts use the TRAIN tasks, more seeds and a higher temperature, e.g.
    --tasks data/tasks_train.jsonl --seeds 0 1 2 3 4 5 6 7 --temperature 0.7
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.harvest import conversations, is_resolved, load_results  # noqa: E402
from gemma_swe.proxy import serve  # noqa: E402
from gemma_swe.serving import (  # noqa: E402
    eval_config, set_seed, start_vllm, stop, submission_adapters, swegemma_cmd, vllm_command, wait_ready)
from gemma_swe.submission_builder import load_yaml  # noqa: E402


def _root_model(sub: Path) -> tuple[str, str | None]:
    cfg = load_yaml(sub / "agent.yaml", sub)
    return str(cfg["model"]).split("/")[-1], cfg.get("adapter")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission-dir", required=True)
    ap.add_argument("--model-dir", required=True, help="served w4a16-ct checkpoint dir (Kaggle Models)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--snapshots-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--vllm-port", type=int, default=8001)
    ap.add_argument("--proxy-port", type=int, default=8000)
    ap.add_argument("--no-serve", action="store_true", help="vLLM already running on --vllm-port")
    ap.add_argument("--route-base-to", default=None,
                    help="rewrite requests for the base model name to this adapter (only if the CLI does not route adapters itself)")
    ap.add_argument("--extra", default="", help="extra swegemma eval args")
    ap.add_argument("--vllm-extra", default="", help="extra vLLM args")
    a = ap.parse_args(argv)

    sub = Path(a.submission_dir)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base, root_adapter = _root_model(sub)
    adapters = submission_adapters(sub)
    proc = None
    if not a.no_serve:
        cmd = vllm_command(a.model_dir, base, adapters, a.vllm_port, extra=a.vllm_extra.split() if a.vllm_extra else None)
        print("vLLM:", " ".join(cmd), flush=True)
        proc = start_vllm(cmd, out / "vllm.log")
    try:
        wait_ready(f"http://127.0.0.1:{a.vllm_port}", proc=proc)
        print("vLLM ready", flush=True)
        model_map = {base: a.route_base_to} if a.route_base_to else {}
        rates = []
        for seed in a.seeds:
            sdir = out / f"submission_seed_{seed}"
            if sdir.exists():
                shutil.rmtree(sdir)
            shutil.copytree(sub, sdir)
            set_seed(sdir, seed, a.temperature)
            log = out / f"proxy_seed_{seed}.jsonl"
            srv = serve(a.proxy_port, f"http://127.0.0.1:{a.vllm_port}", str(log), model_map)
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            rdir = out / f"seed_{seed}"
            cmd = swegemma_cmd(a.tasks, a.snapshots_dir, str(sdir), str(rdir), a.concurrency, eval_config(sdir),
                               a.extra.split() if a.extra else None)
            print(f"[seed {seed}]", " ".join(cmd), flush=True)
            t0 = time.time()
            p = subprocess.run(cmd)
            srv.shutdown()
            srv.server_close()
            res = load_results(rdir)
            n = len(res)
            k = sum(is_resolved(r) for r in res.values())
            rate = k / n if n else 0.0
            rates.append(rate)
            convs = conversations(log) if log.exists() else {}
            reqs = [len(v) for v in convs.values()]
            lat = [r.get("latency", 0) for v in convs.values() for r in v]
            info = {"seed": seed, "resolved": k, "n": n, "rate": round(rate, 4), "exit": p.returncode,
                    "wall_min": round((time.time() - t0) / 60, 1),
                    "mean_requests_per_task": round(statistics.mean(reqs), 1) if reqs else None,
                    "mean_latency_s": round(statistics.mean(lat), 2) if lat else None,
                    "adapter_used": sorted({r["request"].get("model") for v in convs.values() for r in v})}
            print(json.dumps(info), flush=True)
            (out / f"seed_{seed}.info.json").write_text(json.dumps(info, indent=2))
        agg = {"seeds": a.seeds, "rates": rates, "mean": statistics.mean(rates),
               "std": statistics.pstdev(rates) if len(rates) > 1 else 0.0, "root_adapter": root_adapter}
        (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
        print("AGGREGATE", json.dumps(agg), flush=True)
    finally:
        stop(proc)


if __name__ == "__main__":
    main()
