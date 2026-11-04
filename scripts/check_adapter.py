#!/usr/bin/env python
"""Validate a PEFT adapter against the served checkpoint, optionally repairing it.

    python scripts/check_adapter.py --adapter out/main_lora --base-ckpt $W4A16_DIR
    python scripts/check_adapter.py --adapter out/main_lora --base-ckpt $W4A16_DIR --fix out/main_lora_fixed

--fix renames LoRA keys to the base checkpoint's module paths, drops tensors vLLM cannot serve
(KV-shared k/v_proj, towers, embeddings), and normalises adapter_config.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.adapter_utils import fix_adapter, plan_key_fixes, validate_adapter  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--fix", default=None, help="write a repaired adapter here")
    a = ap.parse_args(argv)
    rep = validate_adapter(a.adapter, a.base_ckpt)
    print(json.dumps(rep, indent=2)[:4000])
    if a.fix:
        plan = plan_key_fixes(a.adapter, a.base_ckpt)
        unmapped = [k for k, v in plan.items() if v is None]
        print(f"key plan: {len(plan)} modules, {sum(1 for k, v in plan.items() if v and v != k)} renamed, {len(unmapped)} unmapped")
        print(json.dumps(fix_adapter(a.adapter, a.base_ckpt, a.fix), indent=2))
        rep = validate_adapter(a.fix, a.base_ckpt)
        print("after fix:", json.dumps({k: rep[k] for k in ("ok", "errors", "modules", "rank")}, indent=2))
    sys.exit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
