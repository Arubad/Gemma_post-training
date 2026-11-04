#!/usr/bin/env python
"""Capture the exact prompt/tools/tool-result encoding the harness sends -> calibration.json.

CPU only, ~2 minutes. Needs the competition wheelhouse installed (swegemma, adk-submission,
google-adk). Runs `swegemma eval` on ONE task against a scripted mock model listening where the
harness expects vLLM (127.0.0.1:8000), then distils the logged requests:

    python scripts/capture_harness_prompt.py --profile profiles/e4b_prompt.yaml \
        --tasks data/tasks_smoke.jsonl --snapshots-dir $COMP/snapshots --out data/calibration.json

If swegemma insists on launching its own vLLM instead of attaching to :8000, pass the right
flag through --extra (see `swegemma eval --help`, printed below on failure).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.calibration import calibration_from_requests  # noqa: E402
from gemma_swe.harvest import conversations  # noqa: E402
from gemma_swe.proxy import serve  # noqa: E402
from gemma_swe.records import read_jsonl  # noqa: E402
from gemma_swe.serving import eval_config, swegemma_cmd  # noqa: E402
from gemma_swe.submission_builder import build_dir, load_profile  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True, help="use a profile WITHOUT adapters (e.g. e4b_prompt)")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--task-id", default=None, help="default: first task in --tasks")
    ap.add_argument("--snapshots-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--extra", default="", help="extra args for swegemma eval (space separated)")
    a = ap.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="capture_"))
    prof = load_profile(a.profile)
    sub = build_dir(prof, work / "submission")
    log = work / "capture.jsonl"
    srv = serve(a.port, None, str(log), mock=True)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()

    tasks = list(read_jsonl(a.tasks))
    task = next(t for t in tasks if t["instance_id"] == a.task_id) if a.task_id else tasks[0]
    tfile = work / "task.jsonl"
    tfile.write_text(json.dumps(task) + "\n")
    cmd = swegemma_cmd(str(tfile), a.snapshots_dir, str(sub), str(work / "results"), 1, eval_config(sub),
                       a.extra.split() if a.extra else None)
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    srv.shutdown()
    print(p.stdout[-3000:])
    if not log.exists() or not log.read_text().strip():
        print(p.stderr[-5000:], file=sys.stderr)
        print("\nNo request reached the mock. `swegemma eval --help`:\n", file=sys.stderr)
        print(subprocess.run(["swegemma", "eval", "--help"], capture_output=True, text=True).stdout, file=sys.stderr)
        sys.exit(1)

    convs = conversations(log)
    key, recs = max(convs.items(), key=lambda kv: len(kv[1]))
    instruction = (sub / "prompts" / "system.md").read_text(encoding="utf-8")
    calib = calibration_from_requests([r["request"] for r in recs], task, instruction)
    calib.extra["capture_requests"] = len(recs)
    calib.extra["tool_message_example"] = next(
        (m for r in recs for m in r["request"]["messages"] if m.get("role") == "tool"), None)
    calib.save(a.out)
    (Path(a.out).with_suffix(".requests.jsonl")).write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(json.dumps({
        "saved": a.out, "requests": len(recs), "tool_content_style": calib.tool_content_style,
        "tools": [t.get("function", {}).get("name") for t in calib.tools or []],
        "system_wrapped": calib.system_prefix is not None,
        "system_head": (calib.system_prompt or "")[:200], "user_head": (calib.user_example or "")[:300],
        "request_keys": calib.extra.get("request_keys"), "chat_template_kwargs": calib.extra.get("chat_template_kwargs"),
    }, indent=2))


if __name__ == "__main__":
    main()
