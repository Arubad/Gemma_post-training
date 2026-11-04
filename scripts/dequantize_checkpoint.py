#!/usr/bin/env python
"""Dequantize the served *-qat-w4a16-ct checkpoint to bf16 for LoRA training (CPU is fine).

    python scripts/dequantize_checkpoint.py --src /kaggle/input/gemma-4/.../gemma-4-e4b-it-qat-w4a16-ct \
        --dst /kaggle/tmp/e4b-bf16 --verify

E4B: ~11.5 GB in -> ~18 GB out, a few minutes. 31B: ~23 GB in -> ~62 GB out (needs disk!).
--verify cross-checks the local int4 unpacker against compressed_tensors' on a few tensors.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.dequant import dequantize_checkpoint, verify_against_library  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    if a.verify:
        print(json.dumps(verify_against_library(a.src), indent=2))
    t0 = time.time()
    info = dequantize_checkpoint(a.src, a.dst)
    info["seconds"] = round(time.time() - t0)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
