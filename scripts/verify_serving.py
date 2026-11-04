#!/usr/bin/env python
"""Serving-side verification against a running vLLM (scripts/serve_vllm.py or local_eval --no-serve).

1. render   : our chat_render prefix == vLLM's own /tokenize of the same OpenAI messages+tools
              (proves train-time rendering == serve-time prompt, incl. tool declarations)
2. adapter  : greedy outputs base vs adapter  (--expect same for a zero adapter, different for random/trained)
3. parser   : multi-line edit_file arguments with braces / quotes / newlines round-trip through the
              gemma4 tool parser (in-process, no server needed); with --data, scans SFT records

    python scripts/verify_serving.py render  --endpoint http://127.0.0.1:8001 --tokenizer $CKPT \
        --model gemma-4-e4b-it-qat-w4a16-ct --record-file data/graph_synth.jsonl --calibration data/calibration.json \
        --profile profiles/e4b_lora.yaml
    python scripts/verify_serving.py adapter --endpoint http://127.0.0.1:8001 --model gemma-4-e4b-it-qat-w4a16-ct --adapter zero_lora --expect same
    python scripts/verify_serving.py parser  --tokenizer $CKPT [--data data/remapped.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gemma_swe.calibration import Calibration  # noqa: E402
from gemma_swe.harness_prompt import Budget  # noqa: E402
from gemma_swe.records import assemble, read_jsonl, strip_train_keys  # noqa: E402
from gemma_swe.submission_builder import TEMPLATE_DIR, load_profile  # noqa: E402


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def _openai_msgs(msgs):
    """Our dict-args messages -> the OpenAI wire format LiteLLM sends (arguments as JSON strings)."""
    out = []
    for m in strip_train_keys(msgs):
        m = json.loads(json.dumps(m))
        for tc in m.get("tool_calls") or []:
            tc["function"]["arguments"] = json.dumps(tc["function"]["arguments"])
        out.append(m)
    return out


def cmd_render(a):
    from transformers import AutoTokenizer

    from gemma_swe.chat_render import _tmpl

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    prof = load_profile(a.profile)
    calib = Calibration.load(a.calibration)
    instr = (TEMPLATE_DIR / prof["root"]["instruction"]).read_text(encoding="utf-8")
    ev = prof["evaluation"]
    budget = Budget(float(ev["max_time_minutes"]), ev.get("max_tool_calls"), ev.get("max_turns"), int(ev.get("timeout_seconds", 300)))
    n_ok = n = 0
    for rec in list(read_jsonl(a.record_file))[: a.n]:
        think = int(prof["sampling"].get("thinking_budget") or 0) > 0
        msgs, tools = assemble(rec, calib, instr, budget, prof["root"]["name"], prof["root"].get("description"),
                               rationale_mode="thinking" if think else "none")
        # compare at the last tool-response boundary (= a real generation prompt)
        cut = max(i for i, m in enumerate(msgs) if m["role"] in ("tool", "user")) + 1
        ours = tok(_tmpl(tok, strip_train_keys(msgs[:cut]), tools, gen=True, enable_thinking=think), add_special_tokens=False)["input_ids"]
        body = {"model": a.model, "messages": _openai_msgs(msgs[:cut]), "tools": tools, "add_generation_prompt": True,
                "chat_template_kwargs": {"enable_thinking": think}}
        theirs = _post(a.endpoint.rstrip("/") + "/tokenize", body)["tokens"]
        n += 1
        if ours == theirs:
            n_ok += 1
            continue
        k = next((i for i, (x, y) in enumerate(zip(ours, theirs)) if x != y), min(len(ours), len(theirs)))
        print(f"MISMATCH {rec.get('id')}: ours={len(ours)} vllm={len(theirs)} first diff at token {k}")
        print("  ours :", repr(tok.decode(ours[max(0, k - 20) : k + 20])))
        print("  vllm :", repr(tok.decode(theirs[max(0, k - 20) : k + 20])))
    print(f"render check: {n_ok}/{n} identical")
    sys.exit(0 if n_ok == n else 1)


def cmd_adapter(a):
    prompts = ["Write a Python function that reverses a linked list.", "What does HTTP status 418 mean?",
               "List three uses of functools.lru_cache."]
    same = 0
    for p in prompts:
        outs = []
        for model in (a.model, a.adapter):
            r = _post(a.endpoint.rstrip("/") + "/v1/chat/completions",
                      {"model": model, "messages": [{"role": "user", "content": p}], "temperature": 0, "max_tokens": 64,
                       "chat_template_kwargs": {"enable_thinking": False}})
            outs.append(r["choices"][0]["message"].get("content"))
        same += outs[0] == outs[1]
        print(json.dumps({"prompt": p, "base": outs[0][:120] if outs[0] else outs[0], "adapter": outs[1][:120] if outs[1] else outs[1]}))
    ok = (same == len(prompts)) if a.expect == "same" else (same < len(prompts))
    print(f"identical outputs: {same}/{len(prompts)} (expected {a.expect}) -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


def _parser_and_request(tok, tools):
    ToolParserManager = None
    for mod in ("vllm.tool_parsers", "vllm.entrypoints.openai.tool_parsers"):
        try:
            ToolParserManager = __import__(mod, fromlist=["ToolParserManager"]).ToolParserManager
            break
        except Exception:
            continue
    if ToolParserManager is None:
        raise SystemExit("vLLM tool parser registry not importable")
    req_cls = None
    for mod in ("vllm.entrypoints.openai.protocol", "vllm.entrypoints.openai.chat_completion.protocol"):
        try:
            req_cls = __import__(mod, fromlist=["ChatCompletionRequest"]).ChatCompletionRequest
            break
        except Exception:
            continue
    parser = ToolParserManager.get_tool_parser("gemma4")(tok)
    req = req_cls(model="x", messages=[{"role": "user", "content": "x"}], tools=tools)
    return parser, req


TRICKY = [
    {"filepath": "pkg/mod.py", "old_string": "def f(x):\n    return {\"a\": x}\n",
     "new_string": "def f(x, y=None):\n    d = {\"a\": x, 'b': [1, 2]}\n    if y is not None:\n        d[\"y\"] = f\"{y!r}\"\n    return d\n"},
    {"filepath": "pkg/re_mod.py", "old_string": "PATTERN = re.compile(r'\\d+')", "new_string": "PATTERN = re.compile(r'(?P<n>\\d+)\\s*}')"},
    {"command": "cat > /tmp/check.py << 'EOF'\nimport json\nprint(json.dumps({'ok': True}))\nEOF\npython /tmp/check.py"},
]


def cmd_parser(a):
    from transformers import AutoTokenizer

    from gemma_swe.tool_schema import default_tools

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    tools = default_tools()
    parser, req = _parser_and_request(tok, tools)
    cases = [("edit_file" if "old_string" in c else "run_command", c) for c in TRICKY]
    if a.data:
        for rec in list(read_jsonl(a.data))[: a.n]:
            for t in rec.get("turns", []):
                for tc in t.get("tool_calls") or []:
                    cases.append((tc["name"], tc["arguments"]))
    bad = 0
    for name, args in cases:
        msgs = [{"role": "user", "content": "x"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": "c0", "type": "function", "function": {"name": name, "arguments": args}}]}]
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False)
        s = text.rfind("<|tool_call>")
        e = text.find("<tool_call|>", s)
        out = text[s : e + len("<tool_call|>")]
        info = parser.extract_tool_calls(out, req)
        got = None
        if getattr(info, "tool_calls", None):
            fn = info.tool_calls[0].function
            got = (fn.name, json.loads(fn.arguments))
        if got != (name, args):
            bad += 1
            if bad <= 5:
                print("ROUND-TRIP FAIL:", name, json.dumps(args)[:200], "->", got and json.dumps(got[1])[:200])
    print(f"parser round-trip: {len(cases) - bad}/{len(cases)} ok")
    sys.exit(0 if bad == 0 else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("render")
    r.add_argument("--endpoint", required=True)
    r.add_argument("--tokenizer", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--record-file", required=True)
    r.add_argument("--calibration", default=None)
    r.add_argument("--profile", required=True)
    r.add_argument("--n", type=int, default=5)
    d = sp.add_parser("adapter")
    d.add_argument("--endpoint", required=True)
    d.add_argument("--model", required=True)
    d.add_argument("--adapter", required=True)
    d.add_argument("--expect", choices=["same", "different"], required=True)
    p = sp.add_parser("parser")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--data", default=None)
    p.add_argument("--n", type=int, default=200)
    a = ap.parse_args(argv)
    {"render": cmd_render, "adapter": cmd_adapter, "parser": cmd_parser}[a.cmd](a)


if __name__ == "__main__":
    main()
