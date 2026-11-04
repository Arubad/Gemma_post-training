"""OpenAI-compatible logging proxy (and scripted mock) to sit between the harness and vLLM.

    harness (ADK/LiteLLM) --> proxy :8000 --> vLLM :8001

Every /v1/chat/completions request body and the (reconstructed, if streamed) response are
appended as one JSON line to the log. These logs are:
  * the calibration source (exact system prompt, user prompt, tool declarations, tool-result
    encoding) -> calibration.json
  * the RFT data source (exact message lists the model saw, in the serving format)

`--mock` answers without an upstream using a scripted tool-call sequence, so the harness prompt
can be captured on a CPU-only notebook in a couple of minutes.

Only the standard library is used (runs anywhere the harness runs).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_LOCK = threading.Lock()


def _append(path: str, rec: dict) -> None:
    line = json.dumps(rec, ensure_ascii=False)
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------------------- mock script
MOCK_SCRIPT = [
    ("run_command", {"command": "git status --short && ls"}),
    ("read_file", {"filepath": "README.md", "start_line": 1, "end_line": 5}),
    ("get_status", {}),
    ("submit_patch", {}),
]


def _mock_reply(body: dict) -> dict:
    n_tool_msgs = sum(1 for m in body.get("messages", []) if m.get("role") == "tool")
    declared = {t.get("function", {}).get("name") for t in body.get("tools") or []}
    script = [s for s in MOCK_SCRIPT if s[0] in declared] or MOCK_SCRIPT[-1:]
    msg: dict = {"role": "assistant", "content": None}
    if n_tool_msgs < len(script):
        name, args = script[n_tool_msgs]
        msg["content"] = f"Mock step {n_tool_msgs + 1}."
        msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:8]}", "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}}]
        finish = "tool_calls"
    else:
        msg["content"] = "Done."
        finish = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_from_reply(reply: dict) -> bytes:
    """Encode a non-streamed reply as a single-delta SSE stream."""
    ch = reply["choices"][0]
    delta = {"role": "assistant"}
    if ch["message"].get("content") is not None:
        delta["content"] = ch["message"]["content"]
    if ch["message"].get("tool_calls"):
        delta["tool_calls"] = [dict(tc, index=i) for i, tc in enumerate(ch["message"]["tool_calls"])]
    chunk = {"id": reply["id"], "object": "chat.completion.chunk", "created": reply["created"], "model": reply["model"],
             "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
    end = dict(chunk, choices=[{"index": 0, "delta": {}, "finish_reason": ch["finish_reason"]}])
    return (f"data: {json.dumps(chunk)}\n\n" f"data: {json.dumps(end)}\n\n" "data: [DONE]\n\n").encode()


# ------------------------------------------------------------------------------- stream merge
def merge_stream(lines: list[bytes]) -> dict:
    """Reconstruct a chat.completion from SSE chunks (content, reasoning, tool_calls by index)."""
    content, reasoning, tcs, finish, usage, meta = [], [], {}, None, None, {}
    for raw in lines:
        s = raw.decode("utf-8", "replace").strip()
        if not s.startswith("data:"):
            continue
        s = s[5:].strip()
        if s == "[DONE]" or not s:
            continue
        try:
            ch = json.loads(s)
        except json.JSONDecodeError:
            continue
        meta = meta or {k: ch.get(k) for k in ("id", "model", "created")}
        if ch.get("usage"):
            usage = ch["usage"]
        for c in ch.get("choices", []):
            d = c.get("delta", {})
            if d.get("content"):
                content.append(d["content"])
            r = d.get("reasoning_content") or d.get("reasoning")
            if r:
                reasoning.append(r)
            for tc in d.get("tool_calls") or []:
                i = tc.get("index", 0)
                cur = tcs.setdefault(i, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    cur["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    cur["function"]["name"] += fn["name"]
                if fn.get("arguments"):
                    cur["function"]["arguments"] += fn["arguments"]
            if c.get("finish_reason"):
                finish = c["finish_reason"]
    msg = {"role": "assistant", "content": "".join(content) or None}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    if tcs:
        msg["tool_calls"] = [tcs[i] for i in sorted(tcs)]
    return {**meta, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": usage}


# ------------------------------------------------------------------------------- server
def make_handler(upstream: str | None, log_path: str, model_map: dict[str, str], mock: bool, timeout: float):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _forward_simple(self, method: str):
            if mock:
                if self.path.rstrip("/").endswith("/models"):
                    data = {"object": "list", "data": [{"id": m, "object": "model"} for m in (list(model_map) or ["mock"])]}
                    return self._send(200, json.dumps(data).encode())
                return self._send(200, b"{}")
            n = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(n) if n else None
            req = urllib.request.Request(upstream + self.path, data=body, method=method,
                                         headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    self._send(r.status, r.read(), r.headers.get("Content-Type", "application/json"))
            except urllib.error.HTTPError as e:
                self._send(e.code, e.read())
            except Exception as e:  # upstream down
                self._send(502, json.dumps({"error": str(e)}).encode())

        def do_GET(self):
            self._forward_simple("GET")

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/chat/completions"):
                return self._forward_simple("POST")
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                return self._send(400, b'{"error":"bad json"}')
            orig_model = body.get("model")
            if orig_model in model_map:
                body["model"] = model_map[orig_model]
            t0 = time.time()
            stream = bool(body.get("stream"))
            rec = {"ts": t0, "path": self.path, "orig_model": orig_model, "request": body}
            if mock:
                reply = _mock_reply(body)
                rec.update(response=reply, latency=0.0)
                _append(log_path, rec)
                if stream:
                    return self._send(200, _sse_from_reply(reply), "text/event-stream")
                return self._send(200, json.dumps(reply).encode())
            data = json.dumps(body).encode()
            req = urllib.request.Request(upstream + self.path, data=data, method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    if stream:
                        self.send_response(r.status)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Transfer-Encoding", "chunked")
                        self.end_headers()
                        lines = []
                        for line in r:
                            lines.append(line)
                            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
                            self.wfile.flush()
                        self.wfile.write(b"0\r\n\r\n")
                        rec.update(response=merge_stream(lines), latency=time.time() - t0, streamed=True)
                    else:
                        out = r.read()
                        self._send(r.status, out)
                        try:
                            rec.update(response=json.loads(out), latency=time.time() - t0)
                        except json.JSONDecodeError:
                            rec.update(response={"raw": out.decode("utf-8", "replace")}, latency=time.time() - t0)
            except urllib.error.HTTPError as e:
                err = e.read()
                self._send(e.code, err)
                rec.update(error={"code": e.code, "body": err.decode("utf-8", "replace")}, latency=time.time() - t0)
            except Exception as e:
                self._send(502, json.dumps({"error": str(e)}).encode())
                rec.update(error={"code": 502, "body": str(e)}, latency=time.time() - t0)
            _append(log_path, rec)

    return H


def serve(port: int, upstream: str | None, log_path: str, model_map: dict[str, str] | None = None,
          mock: bool = False, timeout: float = 600.0, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(upstream.rstrip("/") if upstream else None, log_path,
                                                          model_map or {}, mock, timeout))
    srv.daemon_threads = True
    return srv


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--upstream", default="http://127.0.0.1:8001")
    ap.add_argument("--log", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--model-map", default="", help="json dict, e.g. '{\"gemma-4-e4b-it-qat-w4a16-ct\": \"main_lora\"}'")
    a = ap.parse_args(argv)
    mm = json.loads(a.model_map) if a.model_map else {}
    srv = serve(a.port, None if a.mock else a.upstream, a.log, mm, a.mock)
    print(f"proxy listening on :{a.port} -> {'MOCK' if a.mock else a.upstream}; log={a.log}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
