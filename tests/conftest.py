import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SPECIALS = ["<bos>", "<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
            "<|channel>", "<channel|>", "<|tool>", "<tool|>", '<|"|>', "<|think|>"]


class FakeGemmaTokenizer:
    """Python port of the control flow of Gemma 4's chat_template.jinja (Jul 2026 revision):

    * assistant message renders: [thought if after last user] -> tool calls -> following tool responses -> content
    * a trailing tool call (no responses yet) is closed with the stop marker <|tool_response>
    * consecutive assistant messages continue the same model turn; non-empty content closes it with <turn|>
    * generation prompt: '<|turn>model\\n' unless the last thing rendered was a tool call/response
      (then nothing, or '<|channel>thought\\n' when thinking is enabled)
    `gen_prefill_empty_thought=True` mimics 12B/31B: the generation prompt (only) pre-fills an empty thought.
    """

    def __init__(self, gen_prefill_empty_thought: bool = False):
        self.prefill = gen_prefill_empty_thought
        self._vocab: dict[str, int] = {}

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        out = ["<bos>"]
        loop = list(messages)
        prev_type = None
        if enable_thinking or tools or (messages and messages[0]["role"] == "system"):
            out.append("<|turn>system\n")
            if enable_thinking:
                out.append("<|think|>\n")
            if messages and messages[0]["role"] == "system":
                out.append(messages[0]["content"].strip())
                loop = messages[1:]
            for t in tools or []:
                out.append(f"<|tool>declaration:{t['function']['name']}<tool|>")
                prev_type = "tool"
            out.append("<turn|>\n")
        last_user = max([i for i, m in enumerate(loop) if m["role"] == "user"], default=-1)
        prev_non_tool = None
        for idx, m in enumerate(loop):
            if m["role"] == "tool":
                continue
            prev_type = None
            role = "model" if m["role"] == "assistant" else m["role"]
            if not (role == "model" and prev_non_tool == "assistant"):
                out.append(f"<|turn>{role}\n")
            th = m.get("reasoning_content")
            if th and idx > last_user:
                out.append(f"<|channel>thought\n{th}\n<channel|>")
            calls = m.get("tool_calls") or []
            for tc in calls:
                fn = tc["function"]
                assert isinstance(fn["arguments"], dict), "template requires mapping arguments"
                out.append(f"<|tool_call>call:{fn['name']}{json.dumps(fn['arguments'], sort_keys=True)}<tool_call|>")
                prev_type = "tool_call"
            tr = False
            if calls:
                for k in range(idx + 1, len(loop)):
                    if loop[k]["role"] != "tool":
                        break
                    out.append(f"<|tool_response>response:{loop[k].get('name')}{{value:{loop[k]['content']}}}<tool_response|>")
                    tr = True
                    prev_type = "tool_response"
            content = (m.get("content") or "").strip()
            out.append(content)
            nxt = next((loop[j]["role"] for j in range(idx + 1, len(loop)) if loop[j]["role"] != "tool"), None)
            continues = role == "model" and nxt == "assistant" and (not calls or tr)
            if prev_type == "tool_call" and not tr:
                out.append("<|tool_response>")
            elif continues:
                pass
            elif not (tr and not content and nxt is None):
                out.append("<turn|>\n")
            prev_non_tool = m["role"]
        if add_generation_prompt:
            if prev_type not in ("tool_response", "tool_call"):
                out.append("<|turn>model\n" + ("<|channel>thought\n<channel|>" if self.prefill and not enable_thinking else ""))
            elif prev_type == "tool_response" and enable_thinking:
                out.append("<|channel>thought\n")
        return "".join(out)

    def _tok(self, text):
        pat = "|".join(re.escape(s) for s in sorted(SPECIALS, key=len, reverse=True))
        toks, offs = [], []
        for m in re.finditer(pat + r"|[\s\S]", text):
            toks.append(m.group())
            offs.append((m.start(), m.end()))
        return toks, offs

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        toks, offs = self._tok(text)
        ids = [self._vocab.setdefault(t, len(self._vocab)) for t in toks]
        d = {"input_ids": ids}
        if return_offsets_mapping:
            d["offset_mapping"] = offs
        return d

    def decode(self, ids, skip_special_tokens=False):
        inv = {v: k for k, v in self._vocab.items()}
        return "".join(inv[i] for i in ids)


@pytest.fixture
def fake_tok():
    return FakeGemmaTokenizer()


@pytest.fixture
def real_tok():
    """Real Gemma 4 tokenizer + chat template if GSWE_TEST_TOKENIZER points at one (else skip)."""
    p = os.environ.get("GSWE_TEST_TOKENIZER")
    if not p or not Path(p).exists():
        pytest.skip("set GSWE_TEST_TOKENIZER to a Gemma 4 tokenizer dir to run real-template tests")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(p)
