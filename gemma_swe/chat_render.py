"""Render chat messages with the model's own chat template and build loss masks.

Why not TRL's assistant_only_loss: Gemma 4's official template has no {% generation %}
markers. Instead, for every assistant message i we render

    prefix_i = template(messages[:i], add_generation_prompt=True)   # what vLLM sends
    upto_i   = template(messages[:i+1])                             # prefix + model output

and require `full.startswith(prefix_i)` (up to generation-only glue, see GLUE_TAILS). The model-generated region is the part of `upto_i`
after `prefix_i` that agrees with the full render, plus the stop token that follows it in the
full render (`<|tool_response>` after a tool call, `<turn|>` after a final answer). This is
exactly what the model must emit at serve time, including Gemma's size-specific quirks (an
empty thought channel pre-filled in the 12B/31B generation prompt is part of prefix_i and is
therefore never labelled).

If thinking is stripped from history by the template, prefix_i would not be a prefix of the
full render; such records raise RenderMismatch and are dropped (we train with thinking off and
put the short rationale in `content`, which the template keeps in history).

The tokenizer is duck-typed: anything with `apply_chat_template(..., tokenize=False)` and
`__call__(text, add_special_tokens=False, return_offsets_mapping=True)` works.
"""

from __future__ import annotations

from dataclasses import dataclass

from .records import strip_train_keys

STOP_MARKERS = ("<|tool_response>", "<turn|>")


class RenderMismatch(ValueError):
    pass


@dataclass
class Rendered:
    text: str
    input_ids: list[int]
    labels_mask: list[int]  # 1 = trained token
    label_spans: list[tuple[int, int]]  # char spans

    @property
    def n_tokens(self) -> int:
        return len(self.input_ids)

    @property
    def n_label_tokens(self) -> int:
        return sum(self.labels_mask)


def _tmpl(tok, msgs, tools, gen: bool, enable_thinking: bool) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=gen)
    if tools:
        kwargs["tools"] = tools
    try:
        return tok.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def _common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:  # binary search on prefix equality (fast for long strings)
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


# Generation-prompt-only glue that may be absent from the re-rendered history (e.g. 12B/31B pre-fill an
# empty thought channel in the generation prompt when thinking is off, but do not re-render it for past
# turns). A divergent prompt tail made only of these is tolerated: the labels still start exactly where
# the model's own tokens begin.
GLUE_TAILS = ("<|channel>thought\n<channel|>", "<|channel>thought\n", "<|channel>thought<channel|>")


SPECIAL_TOKENS = ("<|turn>", "<turn|>", "<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
                  "<|channel>", "<channel|>", "<|tool>", "<tool|>", '<|"|>', "<|think|>", "<bos>")


def _snap(a: str, b: str, pos: int) -> int:
    """If the common prefix of a and b ends inside a special token (e.g. both continue with '<|' but one
    is <|channel> and the other <|tool_call>), move the divergence point back to that token's start."""
    j = a.rfind("<", max(0, pos - 20), pos)
    if j != -1 and ">" not in a[j:pos] and any(x.startswith(t, j) for x in (a, b) for t in SPECIAL_TOKENS):
        return j
    return pos


def _is_glue(tail: str) -> bool:
    t = tail
    for g in GLUE_TAILS:
        t = t.replace(g, "")
    return t.strip() == ""


def label_spans(tok, messages: list[dict], tools: list[dict] | None, enable_thinking: bool = False) -> tuple[str, list[tuple[int, int]]]:
    clean = strip_train_keys(messages)
    full = _tmpl(tok, clean, tools, gen=False, enable_thinking=enable_thinking)
    spans: list[tuple[int, int]] = []
    for i, m in enumerate(messages):
        if m.get("role") != "assistant" or not m.get("train", True):
            continue
        prefix = _tmpl(tok, clean[:i], tools, gen=True, enable_thinking=enable_thinking)
        start = _snap(prefix, full, _common_prefix_len(prefix, full))
        if start < len(prefix) and not _is_glue(prefix[start:]):
            raise RenderMismatch(
                f"history at assistant turn {i} diverges from the serve-time prompt: "
                f"prompt tail {prefix[start:start + 60]!r} vs render {full[start:start + 60]!r}")
        upto = _tmpl(tok, clean[: i + 1], tools, gen=False, enable_thinking=enable_thinking)
        end = _snap(upto, full, _common_prefix_len(upto, full))
        if end <= start:
            raise RenderMismatch(f"empty completion at assistant turn {i}")
        k = full.rfind("<turn|>", start, end)
        if k != -1:
            end = k + len("<turn|>")  # generation stops at <turn|>; trailing "\n" is template glue
        else:
            for mk in STOP_MARKERS:
                if full.startswith(mk, end):
                    end += len(mk)
                    break
        spans.append((start, end))
    return full, spans


def render_and_mask(tok, messages: list[dict], tools: list[dict] | None, enable_thinking: bool = False) -> Rendered:
    text, spans = label_spans(tok, messages, tools, enable_thinking)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = list(enc["offset_mapping"])
    mask = [0] * len(ids)
    j = 0
    for k, (s, e) in enumerate(offs):
        while j < len(spans) and spans[j][1] <= s:
            j += 1
        if j < len(spans) and spans[j][0] <= s < spans[j][1] and e > s:
            mask[k] = 1
    return Rendered(text, ids, mask, spans)


def decode_labels(tok, r: Rendered) -> list[str]:
    """Human-readable labelled segments, for eyeballing masks in the verification notebook."""
    segs, cur = [], []
    for t, m in zip(r.input_ids, r.labels_mask):
        if m:
            cur.append(t)
        elif cur:
            segs.append(tok.decode(cur, skip_special_tokens=False))
            cur = []
    if cur:
        segs.append(tok.decode(cur, skip_special_tokens=False))
    return segs
