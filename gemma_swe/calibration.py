"""Calibration = the exact strings the scorer sends, captured from a real harness run.

`scripts/capture_harness_prompt.py` runs `swegemma eval` on one task against the mock server in
`gemma_swe.proxy` and writes calibration.json with:

  * system_prompt      exact system message the root agent sends (instruction + ADK identity)
  * user_example       exact first user message for the captured task
  * example_problem    that task's problem statement (to template other tasks into it)
  * example_repo       that task's repo
  * tools              exact OpenAI `tools` list sent to vLLM
  * tool_content_style "raw" (content == tool JSON) or "adk_result_wrap" (content == {"result": "<tool JSON>"})

Every training record is rendered through this calibration so train == serve.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .harness_prompt import Budget, adk_system_prompt, build_user_prompt
from .tool_schema import default_tools, tool_names

_LAYOUT_HDR = re.compile(r"^#+ *Workspace Layout.*$", re.M)


@dataclass
class Calibration:
    system_prompt: str | None = None
    system_prefix: str | None = None  # text ADK puts before agent.instruction
    system_suffix: str | None = None  # text ADK puts after it (identity instruction)
    user_example: str | None = None
    example_problem: str | None = None
    example_repo: str | None = None
    tools: list[dict] | None = None
    tool_content_style: str = "adk_result_wrap"
    source: str = "reconstructed"
    extra: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- io
    @classmethod
    def load(cls, path: str | Path | None) -> "Calibration":
        if not path or not Path(path).exists():
            return cls()
        d = json.loads(Path(path).read_text())
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @property
    def is_captured(self) -> bool:
        return self.source == "captured"

    # ---------------------------------------------------------------- render helpers
    def get_tools(self) -> list[dict]:
        return self.tools if self.tools else default_tools()

    def get_system(self, instruction: str, agent_name: str = "coder", description: str | None = None) -> str:
        if self.system_prefix is not None and self.system_suffix is not None:
            return self.system_prefix + instruction.strip() + self.system_suffix
        if self.system_prompt:
            return self.system_prompt
        return adk_system_prompt(instruction, agent_name, description)

    def get_user(
        self,
        repo: str,
        problem: str,
        budget: Budget,
        layout: list[str] | None,
        hints: str | None = None,
    ) -> str:
        if self.user_example and self.example_problem and self.example_problem in self.user_example:
            return fill_user_template(
                self.user_example, self.example_problem, self.example_repo or "", repo, problem, layout
            )
        return build_user_prompt(
            repo, problem, budget, layout, hints=hints, declared_tools=set(tool_names(self.get_tools()))
        )

    def wrap_tool_content(self, envelope: dict | str) -> str:
        """Serialize a tool result exactly like ADK -> LiteLLM does."""
        raw = envelope if isinstance(envelope, str) else json.dumps(envelope, ensure_ascii=False)
        if self.tool_content_style == "adk_result_wrap":
            return json.dumps({"result": raw}, ensure_ascii=False)
        return raw


def fill_user_template(
    template: str, old_problem: str, old_repo: str, repo: str, problem: str, layout: list[str] | None
) -> str:
    head, _, tail = template.partition(old_problem)
    if old_repo:
        head = head.replace(old_repo, repo)
    m = _LAYOUT_HDR.search(tail)
    if m:
        before = tail[: m.start()]
        tail = before + m.group(0) + "\n" + "\n".join(layout) if layout else before.rstrip()
    return head + problem + tail


def detect_tool_content_style(content: str) -> str:
    try:
        d = json.loads(content)
    except Exception:
        return "raw"
    if isinstance(d, dict) and set(d) == {"result"} and isinstance(d["result"], str):
        try:
            json.loads(d["result"])
            return "adk_result_wrap"
        except Exception:
            return "adk_result_wrap"
    return "raw"


def calibration_from_requests(
    requests: list[dict], task: dict | None = None, instruction: str | None = None
) -> Calibration:
    """Build a Calibration from proxy-logged chat-completions request bodies (one task)."""
    if not requests:
        raise ValueError("no requests captured")
    first = requests[0]
    msgs = first.get("messages", [])
    sys_msg = next((m for m in msgs if m.get("role") in ("system", "developer")), None)
    user_msg = next((m for m in msgs if m.get("role") == "user"), None)

    def _text(m):
        if m is None:
            return None
        c = m.get("content")
        if isinstance(c, list):
            return "".join(p.get("text", "") for p in c if isinstance(p, dict))
        return c

    style = "raw"
    for r in requests:
        for m in r.get("messages", []):
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                style = detect_tool_content_style(m["content"])
                break
        else:
            continue
        break

    sys_text = _text(sys_msg)
    prefix = suffix = None
    if sys_text and instruction and instruction.strip() in sys_text:
        i = sys_text.index(instruction.strip())
        prefix, suffix = sys_text[:i], sys_text[i + len(instruction.strip()):]

    return Calibration(
        system_prompt=sys_text,
        system_prefix=prefix,
        system_suffix=suffix,
        user_example=_text(user_msg),
        example_problem=(task or {}).get("problem_statement"),
        example_repo=(task or {}).get("repo"),
        tools=first.get("tools"),
        tool_content_style=style,
        source="captured",
        extra={
            "system_content_is_list": isinstance((sys_msg or {}).get("content"), list),
            "request_keys": sorted(first.keys()),
            "model": first.get("model"),
            "sampling": {k: first.get(k) for k in ("temperature", "top_p", "top_k", "max_tokens", "seed") if k in first},
            "chat_template_kwargs": first.get("chat_template_kwargs"),
        },
    )
