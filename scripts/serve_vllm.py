#!/usr/bin/env python
"""Start vLLM with the scorer's exact flags (+ optional adapters) and block until ready.

    python scripts/serve_vllm.py --model-dir $CKPT --served-name gemma-4-e4b-it-qat-w4a16-ct \
        --adapter zero_lora=/kaggle/tmp/zero_lora --adapter rand_lora=/kaggle/tmp/rand_lora --port 8001 &

Prints the PID; stop with `kill -TERM -<pid>` (process group). Log: <log> (default vllm.log).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.serving import start_vllm, vllm_command, wait_ready  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--served-name", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="NAME=PATH")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--log", default="vllm.log")
    ap.add_argument("--extra", default="")
    a = ap.parse_args(argv)
    adapters = dict(x.split("=", 1) for x in a.adapter)
    cmd = vllm_command(a.model_dir, a.served_name, adapters, a.port, extra=a.extra.split() if a.extra else None)
    print(" ".join(cmd), flush=True)
    p = start_vllm(cmd, a.log)
    wait_ready(f"http://127.0.0.1:{a.port}", proc=p)
    print(f"READY pid={p.pid}", flush=True)


if __name__ == "__main__":
    main()
