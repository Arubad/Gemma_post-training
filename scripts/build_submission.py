#!/usr/bin/env python
"""Build + validate + zip a submission from a profile.

    python scripts/build_submission.py --profile profiles/e4b_lora.yaml \
        --adapter main_lora=/kaggle/working/adapters/main_lora_fixed \
        --base-ckpt /kaggle/input/gemma-4/.../gemma-4-e4b-it-qat-w4a16-ct \
        --dir /kaggle/working/submission --zip /kaggle/working/submission.zip

--base-ckpt (optional) checks every LoRA module name against the served checkpoint so an adapter
that vLLM would silently ignore is caught here. Exit code 1 on any validation error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.submission_builder import build_dir, load_profile, validate_dir, zip_dir  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--adapter", action="append", default=[], help="NAME=PATH (overrides the profile)")
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--zip", default=None)
    a = ap.parse_args(argv)
    adapters = dict(x.split("=", 1) for x in a.adapter)
    prof = load_profile(a.profile)
    out = build_dir(prof, a.dir, adapters)
    rep = validate_dir(out, a.base_ckpt)
    print(json.dumps({k: v for k, v in rep.items() if k != "adapters"}, indent=2))
    for n, r in rep["adapters"].items():
        print(f"adapter {n}: rank={r['rank']} modules={r['modules']} size={r['size_mb']}MB ok={r['ok']}")
    if not rep["ok"]:
        sys.exit(1)
    if a.zip:
        z = zip_dir(out, a.zip)
        print(f"wrote {z} ({z.stat().st_size / 2**20:.1f} MB)")
    print("files:", *sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()), sep="\n  ")


if __name__ == "__main__":
    main()
