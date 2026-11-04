import json
import threading
import urllib.request

from gemma_swe.harvest import chains, conversations, harvest, mark_train_flags
from gemma_swe.proxy import merge_stream, serve


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def test_mock_proxy_and_harvest(tmp_path):
    log = tmp_path / "log.jsonl"
    srv = serve(0, None, str(log), mock=True)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    tools = [{"type": "function", "function": {"name": n}} for n in ("run_command", "read_file", "get_status", "submit_patch")]
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "You are evaluating ... Problem Statement:\nBug in foo()"}]
    try:
        for step in range(5):
            stream = step == 2
            raw = _post(port, {"model": "m", "messages": msgs, "tools": tools, "stream": stream})
            reply = merge_stream(raw.splitlines(keepends=True)) if stream else json.loads(raw)
            m = reply["choices"][0]["message"]
            msgs.append(m)
            if not m.get("tool_calls"):
                break
            tc = m["tool_calls"][0]
            res = {"status": "error", "error_type": "FileEditError"} if step == 1 else {"status": "ok"}
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["function"]["name"],
                         "content": json.dumps({"result": json.dumps(res)})})
    finally:
        srv.shutdown()
    names = [m["tool_calls"][0]["function"]["name"] for m in msgs if m.get("tool_calls")]
    assert names == ["run_command", "read_file", "get_status", "submit_patch"]
    convs = conversations(log)
    assert len(convs) == 1
    (recs,) = convs.values()
    ch = chains(recs)
    assert len(ch) == 1 and ch[0]["n_requests"] == 5 and ch[0]["messages"][-1]["content"] == "Done."
    flags = [m["train"] for m in mark_train_flags(ch[0]["messages"]) if m["role"] == "assistant"]
    assert flags == [True, False, True, True, True]  # read_file turn errored -> masked
    rdir = tmp_path / "res"
    rdir.mkdir()
    (rdir / "task_results.jsonl").write_text(json.dumps({"instance_id": "foo_1", "resolved": True}) + "\n")
    out = harvest(log, rdir, [{"instance_id": "foo_1", "problem_statement": "Bug in foo()"}], "run")
    assert len(out) == 1 and out[0]["format"] == "exact" and out[0]["meta"]["resolved"]


def test_chain_break_on_compaction():
    sys_, usr = {"role": "system", "content": "S"}, {"role": "user", "content": "U"}
    a1 = {"role": "assistant", "content": "a1", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "x", "arguments": "{}"}}]}
    t1 = {"role": "tool", "tool_call_id": "1", "content": "r1"}
    resp = lambda c: {"choices": [{"message": {"role": "assistant", "content": c}}]}  # noqa: E731
    recs = [
        {"ts": 1, "request": {"messages": [sys_, usr]}, "response": {"choices": [{"message": a1}]}},
        {"ts": 2, "request": {"messages": [sys_, usr, a1, t1]}, "response": resp("a2")},
        {"ts": 3, "request": {"messages": [sys_, usr, {"role": "user", "content": "SUMMARY"}]}, "response": resp("a3")},
    ]
    ch = chains(recs)
    assert len(ch) == 2 and ch[0]["messages"][-1]["content"] == "a2" and ch[1]["messages"][-1]["content"] == "a3"
