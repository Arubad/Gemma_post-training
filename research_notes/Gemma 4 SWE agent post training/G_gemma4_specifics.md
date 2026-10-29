# Cluster G: Gemma 4 model-family specifics for fine-tuning a tool-using coding agent

Research date: 2026-10-29. Primary sources: Gemma 4 Technical Report (arXiv 2607.02770v2, Jul 2026), ai.google.dev Gemma 4 docs (model card, prompt formatting, function calling, thinking), HF `google/gemma-4-*` repos (config.json, chat_template.jinja, HF API file sizes; all last modified 2026-07-20), vLLM recipe + parser docs, Unsloth Gemma 4 guide, GitHub issues (transformers, prime-rl, NeMo Automodel, OpenCode, gemma-agent-lab), and the local `HARNESS_README.md` / `HANDOFF.md` for competition constraints. Anything marked **[unverified]** could not be confirmed against a primary source.

## 1. Architecture per size (E2B / E4B / 12B / 31B)

### Takeaway
All Gemma 4 sizes are decoder-only transformers with interleaved local sliding-window (RoPE 10k) and global (p-RoPE 1M, partial_rotary_factor 0.25) attention in a 5:1 pattern (4:1 for E2B), QK-norm, pre+post RMSNorm, `final_logit_softcapping=30`, a 262,144-token vocab, and a frozen, separately named vision tower (encoder-free projection for 12B) that LoRA can simply skip. Global layers in 12B/31B have head_dim 512 with K=V (`attention_k_eq_v=true`) and very few KV heads (31B: 4; 12B: 1), which is the binding constraint for tensor parallelism and breaks FlashAttention-2.

### Cited Findings
Table assembled from the technical report Table 1 ([arXiv 2607.02770](https://arxiv.org/abs/2607.02770)), the [model card](https://ai.google.dev/gemma/docs/core/model_card_4), and the HF `config.json` files ([31B](https://huggingface.co/google/gemma-4-31B-it/raw/main/config.json), [12B](https://huggingface.co/google/gemma-4-12B-it/raw/main/config.json), [E4B](https://huggingface.co/google/gemma-4-E4B-it/raw/main/config.json)).

| Property | E2B | E4B | 12B ("Unified") | 31B |
|---|---|---|---|---|
| Effective / total params | 2.3B / 5.0B (card: 5.1B) | 4.5B / 8.0B | 11.95B (text einsums 10.89B + 1.0B embedder) | 30.7B (29.29B + 1.41B embedder) |
| Per-layer embeddings (PLE) | yes (400M) | yes (670M; `hidden_size_per_layer_input=256`, `vocab_size_per_layer_input=262144`) | no | no |
| Layers | 35 | 42 (35 sliding / 7 full) | 48 (40 sliding / 8 full) | 60 (50 sliding / 10 full; full at layers 6,12,...,60) |
| hidden / intermediate | not fetched | 2560 / 10240 | 3840 / 15360 | 5376 / 21504 |
| Q heads / local KV / global KV | not fetched | 8 / 2 / (null) | 16 / 8 / 1 | 32 / 16 / 4 |
| head_dim local / global | not fetched | 256 / 512 | 256 / 512 | 256 / 512 |
| `attention_k_eq_v` (global) | false (report: "except E2B/E4B") | false | true | true |
| KV-cache sharing (`num_kv_shared_layers`) | 20 of 35 | 18 of 42 | 0 | 0 |
| Sliding window | 512 | 512 | 1024 | 1024 |
| Local:global ratio | 4:1 | 5:1 | 5:1 | 5:1 |
| Max context (`max_position_embeddings`) | 128K | 131,072 | 262,144 | 262,144 |
| `final_logit_softcapping` | 30 (E4B value; E2B assumed) | 30.0 | 30.0 | 30.0 |
| Vision | 150M ViT (16 layers), frozen | 150M ViT | encoder-free: 48x48 patch matmul (35M) -> `gemma4_unified_vision` | 550M ViT (27 layers, d=1152), frozen |
| Audio | 305M USM conformer (12 layers) | 305M | encoder-free 640-d projection | none |
| HF architecture class | `Gemma4ForConditionalGeneration` | `Gemma4ForConditionalGeneration` | `Gemma4UnifiedForConditionalGeneration` | `Gemma4ForConditionalGeneration` |
| bf16 weights on disk (text+vision) | 4.6 GB (report Table 3) | 9.0 GB | 24.0 GB | 64.0 GB (HF safetensors: 62.55 GB, 2 shards) |
| Q4_0 / mobile quantized | 0.8 GB (mobile) | 2.3 GB (mobile) | 7.65 GB | 19.2 GB |
| int8 KV cache @32k | +0.05 GB | +0.14 GB | +0.28 GB | +1.10 GB |

- Tokenizer is the Gemini SentencePiece tokenizer (split digits, preserved whitespace, byte fallback), 262k vocab; pre-training cutoff January 2025; all models Apache 2.0 — [arXiv 2607.02770](https://arxiv.org/abs/2607.02770).
- Global layers use p-RoPE p=0.25 at 1M base; local layers RoPE 10k; K reused as V in global layers (not in E2B/E4B); together these cut global KV cache 37.5% — [arXiv 2607.02770](https://arxiv.org/abs/2607.02770).
- 31B TP constraint: `num_global_key_value_heads=4` makes tp=4 the maximum integer split; tp=8 fails at sharding — [arXiv 2605.25645](https://arxiv.org/abs/2605.25645) (TPU fine-tuning paper, Table 3). For 12B (1 global KV head) vLLM must replicate KV heads at TP4; the competition harness additionally passes `--hf-overrides '{"architectures":["Gemma4ForCausalLM"],"model_type":"gemma4"}'` for `gemma-4-12b-it-qat-w4a16-ct` — local `HARNESS_README.md` section 3.
- HF module names for LoRA: `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`; the vision tower contains identically named projections so a regex exclusion (`^(?!.*vision)`) is required — [arXiv 2605.25645](https://arxiv.org/abs/2605.25645) section 3.3. Unsloth's text recipe trains language attention+MLP only with vision layers off — [Unsloth guide](https://unsloth.ai/docs/models/gemma-4/train). A community adapter card reports Gemma 4 wraps vision projections in `Gemma4ClippableLinear`, which PEFT refuses to target, so LoRA targets must be resolved from the live module tree excluding the vision tower **[unverified: community model card]** — [HF search result](https://huggingface.co/karthik-a/sparkflashxgemma4).
- Report: vision and audio encoders are frozen during pre-training; 12B's encoder-free design "alleviates the need for separate encoders" — [arXiv 2607.02770](https://arxiv.org/abs/2607.02770) section 2.3.

### Inferences
- For text-only SFT the vision tower can be excluded from LoRA targets and left loaded (150M-550M bf16 params = 0.3-1.1 GB), or the model loaded as `Gemma4ForCausalLM` (the harness already uses this override for 12B at serve time). Dropping it from the adapter is both safe and standard.
- On E2B/E4B, LoRA on `v_proj` in KV-shared layers (`v_proj=None` on some layers per transformers PR #45253) may silently skip modules; inspect PEFT's targeted-module count.
- Per-layer-embedding tables (670M on E4B) are embeddings, not linears; `target_modules="all-linear"` will not touch them, which is desirable.

### Gaps
- E2B `config.json` was not fetched (heads, hidden size); table cells say "not fetched".
- No official statement found on whether the 12B `Gemma4ForCausalLM` override is supported for training in transformers (only the harness uses it for vLLM).

## 2. Chat / tool / thinking format, and what vLLM's parsers expect

### Takeaway
Gemma 4 uses `<|turn>role\n ... <turn|>` turns (roles `system|user|model`), `<|think|>` at the top of the system turn to enable thinking, `<|channel>thought\n...<channel|>` for reasoning, `<|tool>declaration:NAME{...}<tool|>` for schemas, `<|tool_call>call:NAME{k:<|"|>v<|"|>}<tool_call|>` for calls and `<|tool_response>response:NAME{...}<tool_response|>` for results, with `<|"|>` as the string delimiter and uppercase schema types. vLLM's `gemma4` reasoning parser strips the `thought\n` label, treats `<|tool_call>` as implicit end-of-reasoning, and pre-initialises reasoning state only when the prompt itself ends in an open `<|channel>` (which the official template produces after a tool response when thinking is on; with thinking off it pre-fills an empty block for 12B/31B). The harness's `thinking_budget` is a token-budget logits processor, not a model-native level.

### Cited Findings
- Literal format (from the [prompt-formatting doc](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4) and report Table 11, [arXiv 2607.02770](https://arxiv.org/abs/2607.02770)):
```
<bos><|turn>system
<|think|>
<|tool>declaration:search_train{description:<|"|>...<|"|>,parameters:{properties:{to:{type:<|"|>STRING<|"|>}},required:[<|"|>to<|"|>],type:<|"|>OBJECT<|"|>}}<tool|><turn|>
<|turn>user
To Rome.<turn|>
<|turn>model
<|channel>thought
...reasoning...<channel|>Looking for tickets:<|tool_call>call:search_train{from:<|"|>Athens<|"|>,to:<|"|>Rome<|"|>}<tool_call|><turn|>
<|turn>model
<|tool_response>response:search_train{results:[...]}<tool_response|>
```
  Add `<bos>` via the tokenizer, never as text; IT models end turns with `<turn|>` (PT uses `<eos>`) and "fine-tuning either model type thus requires adding their respective end tokens" — [arXiv 2607.02770](https://arxiv.org/abs/2607.02770) section 3.
- Tool declarations are not JSON: brace-delimited, keys sorted alphabetically, types uppercased (`STRING`, `OBJECT`, `ARRAY`), enums as lists; the doc's reference regex for calls is `<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>`; numbers/booleans are unwrapped; `<|tool_response>` doubles as a stop sequence — [function-calling doc](https://ai.google.dev/gemma/docs/capabilities/text/function-calling-gemma4).
- HF `chat_template.jinja` (31B-it): `assistant` -> `model`; a system turn is emitted if `enable_thinking`, `tools`, or a system/developer first message exists; reasoning (`reasoning`/`reasoning_content`) is rendered only for messages after the last user message or when `preserve_thinking` is set and the message has tool calls; `strip_thinking` removes `<|channel>...<channel|>` from historical model content; tool results are accepted as legacy `tool_responses` on the assistant message or as `role: tool` messages matched by `tool_call_id`; string-typed `arguments` raise an exception; generation prompt: `<|turn>model\n` plus, if thinking is disabled, `<|channel>thought\n<channel|>`; if the last message is a tool response and thinking is enabled, `<|channel>thought\n`; if the last is a tool response with thinking off, or a tool call, nothing is appended — [chat_template.jinja](https://huggingface.co/google/gemma-4-31B-it/raw/main/chat_template.jinja).
- Size-dependent quirk: 12B/26B/31B emit a thought channel even when thinking is off, so the template pre-fills an empty one; E2B/E4B do not — [thinking doc](https://ai.google.dev/gemma/docs/capabilities/thinking); [model card](https://ai.google.dev/gemma/docs/core/model_card_4).
- Multi-turn rule: strip prior-turn thoughts, except within a single model turn containing function calls, where thoughts between calls must be kept — [prompt-formatting doc](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4).
- No native `thinking_budget`/`thinking_level`: Google documents thinking as boolean only; a "LOW" instruction in the system prompt is a proof-of-concept that cut thinking tokens ~20% — [prompt-formatting doc](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4). vLLM maps `reasoning_effort` low/medium/high -> `enable_thinking=true`, `none` -> false — [vLLM reasoning docs](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html). The competition harness forwards `thinking_config.thinking_budget` (0..32768, default 4096) as `extra_body.thinking_token_budget` to vLLM's `ThinkingTokenBudgetLogitsProcessor`; 0 sets `enable_thinking:false`; `thinking_level` should be omitted with vLLM+LoRA — local `HARNESS_README.md` section 2.
- vLLM flags: `--enable-auto-tool-choice --reasoning-parser gemma4 --tool-call-parser gemma4 --chat-template examples/tool_chat_template_gemma4.jinja`; per-request `chat_template_kwargs: {"enable_thinking": true}`; server default via `--default-chat-template-kwargs` — [vLLM Gemma 4 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Google/Gemma4.html).
- vLLM `gemma4` reasoning parser internals: strips the `thought\n` prefix; `_prompt_ends_in_open_reasoning` scans prompt tokens backwards and pre-sets REASONING state only if `<|channel>` appears before any closing/tool-call/turn/tool-response boundary; a prompt that merely opens a new turn "must not pre-initialise reasoning" because output without markers is classified as content; `<|tool_call>` is an implicit end of reasoning; empty reasoning -> `None`; the parser forces `skip_special_tokens=False`; streaming tool-arg parsing withholds partial `<|"|>` strings — [vLLM parser docs](https://docs.vllm.ai/en/stable/api/vllm/parser/gemma4/). The tool parser is accumulate-then-parse-then-diff since the format is not JSON; a separate `Gemma4EngineToolParser` skips JSON structured output for named/required tool choice — [vLLM API docs](https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/gemma4_tool_parser/).
- Early bug: with vLLM 0.19.0 `reasoning` came back `None` and thoughts leaked into `content` because special tokens were stripped; fixed by vLLM PR #39027 (on main by Apr 10 2026) — [HF discussion #28](https://huggingface.co/google/gemma-4-31B-it/discussions/28). Dynamo found that with thinking off, Gemma 4 leaks empty channel markers into content on the turn right after a tool result unless the reasoning parser runs alongside the tool parser — [ai-dynamo PR #17486](https://github.com/ai-dynamo/dynamo/pull/17486).
- Template history: the official template was revised (published 2026-07-09, HF updated 2026-07-15) to suppress `<turn|>` between consecutive assistant messages (linked to repeated tool-call loops), add `preserve_thinking`, open `<|channel>thought` after a tool response when thinking is enabled, render null args, and raise on string arguments — [note.com template diff](https://note.com/unco3/n/ne5d382e3287b?hl=en). Earlier (Apr 16-17) OpenCode debugging found two root causes of loops: the system prompt sent as a content-part list rendered as a JSON array, and empty `content` on tool-call messages causing a premature `<turn|>` — [OpenCode issue #21034](https://github.com/anomalyco/opencode/issues/21034). A community HF thread (Apr 9) complained the template is "too complicated", citing the empty thought channel and "multiple useless tool callings"; no Google reply — [HF discussion #53](https://huggingface.co/google/gemma-4-31B-it/discussions/53).

### Inferences
- Train/serve consistency is the main risk: the serving prompt for 12B/31B with thinking off ends in `<|turn>model\n<|channel>thought\n<channel|>`, so training targets for non-thinking turns should begin right after that prefix (the empty block is prompt, not label). After a tool response with thinking off nothing is appended, yet the model natively emits the empty block (Dynamo observation) — the training render at that position must match whatever the template produces (open question section 6).
- If you SFT with thinking on, training rows must contain `<|channel>thought\n...<channel|>` before content/tool calls, and historical assistant turns must have thoughts stripped except within tool-call chains (`preserve_thinking=True`), exactly mirroring the template; otherwise the model learns to think at positions where the serving prompt will not open a channel.
- Because the harness enforces a `thinking_token_budget` logits processor (default 4096), thinking traces in training data should be short (under ~2-3k tokens) or the model will be force-closed mid-thought at inference.
- Tool-call strings containing `}` (code patches) collide with the doc's non-greedy regex `\{(.*?)\}`; vLLM's parser is a proper state machine, but how it handles `}` or newlines inside `<|"|>...<|"|>` in large patch arguments is unverified — test with multi-line code arguments early.

### Gaps
- Could not fetch the vLLM `gemma4_tool_parser.py` source directly (404 on guessed paths); only the docs summary was available. Exact handling of nested braces, escaped `<|"|>` inside strings, and parallel calls is unverified.
- No official statement on whether the model is trained to emit multiple `<|tool_call>` blocks in one turn (docs only show the parser returning a list).

## 3. Benchmark baselines (agentic coding, tool use), QAT vs bf16

### Takeaway
Google reports strong coding (LiveCodeBench v6 80.0 / 72.0 / 52.0 / 44.0 for 31B / 12B / E4B / E2B) and tool-use (tau2 avg 76.9 / 69.0 / 42.2 / 24.5) numbers, but Terminal-Bench Hard is weak (36 / 18 / 8 / 3) and no SWE-bench Verified or BFCL numbers exist from Google. No quantitative QAT-vs-bf16 deltas were published anywhere official.

### Cited Findings
From Table 5 of [arXiv 2607.02770](https://arxiv.org/abs/2607.02770) (all thinking mode; Gemma 3 27B non-thinking):

| Benchmark | 31B | 26B-A4B | 12B | E4B | E2B | Gemma 3 27B |
|---|---|---|---|---|---|---|
| LiveCodeBench v6 | 80.0 | 77.1 | 72.0 | 52.0 | 44.0 | 29.1 |
| Codeforces Elo | 2150 | 1718 | 1659 | 940 | 633 | 110 |
| SciCode | 43.0 | 40.0 | 38.0 | 24.0 | 21.0 | 21.0 |
| Terminal Bench Hard | 36.0 | 14.0 | 18.0 | 8.0 | 3.0 | 4.0 |
| Tau2 airline / retail / telecom | 75.0 / 86.4 / 69.3 | 76.0 / 85.5 / 43.0 | 75.0 / 77.6 / 54.4 | 52.0 / 67.1 / 18.4 | 31.0 / 34.6 / 19.7 | 39.0 / 6.6 / 3.1 |
| IFEval / IFBench | 98.9 / 76.0 | 98.5 / 72.0 | 97.2 / 74.0 | 96.7 / 44.0 | 94.6 / 38.0 | 90.4 / 32.0 |
| MRCR v2 8-needle 128k | 66.4 | 44.1 | 43.4 | 25.4 | 19.1 | 13.5 |
| RULER 32k / 128k (no thinking) | 96.8 / 96.4 | 97.3 / 89.8 | 96.4 / 91.2 | 95.2 / 86.6 | 83.0 / 70.4 | 91.1 / 66.0 |

- The model card confirms the same numbers and reports Tau2 averages 76.9 / 68.2 / 69.0 / 42.2 / 24.5; it lists no SWE-bench Verified, BFCL or Terminal-Bench — [model card](https://ai.google.dev/gemma/docs/core/model_card_4); same on the [HF 31B-it card](https://huggingface.co/google/gemma-4-31B-it).
- Arena Text (June 19 2026): Gemma 4 31B Elo 1451 (rank 43, top dense open model), 26B-A4B 1438, Gemma 3 27B 1366 — [arXiv 2607.02770](https://arxiv.org/abs/2607.02770) Table 4.
- QAT quality: the report says QAT reduces footprint "with minimal impact on quality"; also "to enable stable inference in fp16, we introduce a scalar scale at each block" — [arXiv 2607.02770](https://arxiv.org/abs/2607.02770) section 2.5. Google's QAT blog (Jun 5 2026) gives only qualitative claims ("even higher overall quality" than PTQ baselines), no bf16 deltas — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/). The w4a16-ct card says QAT "allows preserving similar quality to bfloat16" with no numbers — [HF w4a16-ct card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct).
- Only third-party quant-quality data found: Unsloth reports KLD/Top-1 for GGUFs (e.g., 26B Unsloth Q4 85.63% Top-1 vs 70.20% for naive Q4_0) — [Unsloth QAT page](https://unsloth.ai/docs/models/gemma-4/qat).
- Hands-on single-task agentic test (Codex CLI, Q4_K_M): 31B solved a Python function + tests 5/5 first try with 3 clean tool calls but used 185k tokens and 7 min on a GB10; 26B-A4B needed 5 attempts, ~10 messy tool calls, and corrupted shell heredocs — [danielvaughan gist](https://gist.github.com/danielvaughan/9c414ce1b49b1940dfc87bb9d7534a55).
- Loop reports: OpenCode users saw repeated commands after successful tool results, infinite loops on any tool-requiring request, 31B "stuck in thinking", and tool calls emitted as plain text (Apr 2026, LM Studio/llama.cpp/vLLM 0.19 + nightly); some improvement with updated quants and the Jul template; the issue was auto-closed Aug 28 with no upstream fix — [OpenCode issue #21034](https://github.com/anomalyco/opencode/issues/21034). HANDOFF notes the Kaggle discussion says `gemma-4-31b-it-qat-w4a16-ct` "has scored well for most teams" — local `HANDOFF.md`.

### Inferences
- Terminal-Bench Hard (closest to multi-step SWE work) collapses from 36 (31B) to 8 (E4B) and 3 (E2B), far steeper than LiveCodeBench; expect the small models' agentic gap to be the dominant limitation and the main thing SFT must fix.
- Tau2-telecom (the hardest, most multi-step split) is where E4B is weakest (18.4) — a proxy for long tool chains.

### Gaps
- No SWE-bench Verified or BFCL for any Gemma 4 size from Google. A ~63% SWE-bench Verified figure for 31B circulates in a third-party blog but could not be traced to any official source — **[unverified, likely not real]**.
- No QAT-vs-bf16 benchmark deltas for the w4a16-ct checkpoints; must be measured on our own eval set.

## 4. Fine-tuning recipe facts and pitfalls

### Takeaway
Community recipes converge on LoRA r=8-32 (alpha >= r), LR 1e-4 to 2e-4 (2e-5 for long runs), bf16, AdamW, warmup + cosine/linear, 1 epoch, `max_grad_norm` 0.3-1.0, vision tower frozen/excluded. The pitfalls are specific and recent: `use_cache=False` (forced by gradient checkpointing) corrupted Gemma 4 attention in transformers <= 5.5 (fixed Apr 9 2026); FlashAttention-2 crashes on head_dim-512 global layers; cuDNN SDPA produced NaNs in bf16 on H100 (NeMo); `final_logit_softcapping` lives in `text_config` and was silently ignored by some trainers; explicit `attn_implementation="sdpa"` fails to load in transformers 5.20 because the audio sub-model lacks SDPA; TRL `assistant_only_loss` needs `{% generation %}` markers the official template lacks; PEFT cannot wrap compressed-tensors (w4a16) modules, so train on dequantized bf16 weights and serve the adapter on the w4a16 base.

### Cited Findings
- Unsloth recommendations: LR 2e-4 (2e-5 long runs); text LoRA r=8, alpha=8, dropout 0, language attention+MLP only; multimodal r=32 all-linear; batch 1 x grad-accum 4; adamw_8bit, wd 0.001; 1 epoch; `max_grad_norm=0.3` in the vision recipe; `dtype=None`; ~1.5x faster / 60% less VRAM vs FA2; VRAM: E2B LoRA 8-10 GB, E4B QLoRA 10 GB / LoRA 17 GB, 31B QLoRA 22 GB, 26B-A4B LoRA >40 GB; `train_on_responses_only(instruction_part="<|turn>user\n", response_part="<|turn>model\n")`; chat templates `gemma-4` vs `gemma-4-thinking`; keep >= 75% reasoning examples if preserving thinking; do not feed earlier thought blocks back into history; E2B/E4B multimodal losses of 13-15 are "normal" vs 1-3 for 26B/31B; T4 fp16 overflows on the audio attention `-1e9` mask — [Unsloth guide](https://unsloth.ai/docs/models/gemma-4/train). Unsloth's 31B Kaggle notebook: https://www.kaggle.com/code/danielhanchen/gemma4-31b-unsloth.
- `use_cache=False` bug: garbage logits and loss 10-15 (vs 2-6) on E2B with transformers 5.5.0 because Gemma 4 needs `past_key_values` internally for KV sharing, `v_proj=None` layers and sliding masks; gradient checkpointing forces `use_cache=False`; fixed in PR #45312, issue closed Apr 9 2026 — [transformers #45242](https://github.com/huggingface/transformers/issues/45242). Unsloth notes the same ("garbage logits") and ships its own fix — [Unsloth guide](https://unsloth.ai/docs/models/gemma-4/train).
- prime-rl 31B SFT+LoRA (Apr 27 2026, 8xH200): FA2 crashes on head_dim 512 global layers -> SDPA on global layers; `final_logit_softcapping` read from the top-level config returned None so logits reached +-122 instead of +-30 and grad norms roughly doubled; pre-clip grad norms 1k-1M vs 15k-80k for HF+PEFT (unresolved); settings used r=16, alpha=32, LR 1e-4, wd 0, max_norm 1.0, cosine, 20 warmup, seq 16384, vision frozen — [prime-rl #2362](https://github.com/PrimeIntellect-ai/prime-rl/issues/2362).
- NeMo Automodel: bf16 training with `attn_implementation="sdpa"` gives NaN gradients on H100 (not A100) via the cuDNN SDPA path for head_dim-512 layers — [Automodel #3232](https://github.com/NVIDIA-NeMo/Automodel/issues/3232). transformers 5.20: explicit `attn_implementation="sdpa"` raises for Gemma 4 because the audio sub-model does not support SDPA; leave it unset so per-sub-model defaults apply — [d1a #324](https://github.com/jonpol01/d1a/issues/324). A model card reports SDPA returned wrong last-position logits at 32k+1 sequence length in transformers 5.20.2 (eval only; training unaffected) **[community]**.
- 31B/26B inference `IndexError`: `num_kv_shared_layers=0` makes `layer_types[:-0]` empty and crashes `Cache.update` — [Unsloth guide](https://unsloth.ai/docs/models/gemma-4/train).
- TRL: `assistant_only_loss=True` requires `{% generation %}`/`{% endgeneration %}` in the template; TRL patches known families (Gemma/Gemma2 listed, no Gemma 4 entry found); tool responses must stay outside the generation block — [TRL chat templates](https://huggingface.co/docs/trl/en/chat_templates); `chat_template_path` crashes with Gemma 4 processors (workaround: pass the `-it` tokenizer as `processing_class`) — [TRL #7775](https://github.com/huggingface/trl/issues/7775); checkpoints whose template lacks markers are rejected at SFTTrainer init — [TRL #7962](https://github.com/huggingface/trl/issues/7962).
- TPU paper's GPU baseline (TRL SFTTrainer + FSDP, 2xH100): LoRA on q/k/v/o/gate/up/down with vision excluded; linear warmup 100 steps then cosine; `max_grad_norm=1.0`; drop (do not truncate) sequences > `max_seq_len`; strips `<think>` CoT from CodeV-R1 and trains on answers only — [arXiv 2605.25645](https://arxiv.org/abs/2605.25645).
- 4xL4 feasibility (gemma-agent-lab): loading `gemma-4-31b-it-qat-w4a16-ct` with `run_compressed=False` dequantises packed W4A16 to bf16 in 217 s; `get_peft_model` on the packed model raises `AttributeError: 'Linear' object has no attribute 'weight'`; 4-rank FSDP OOMed host RAM (each rank materialises a ~286 GiB replica vs 189 GiB RAM); single-process `device_map` across 4 L4s peaked 7.6 / 18.5 / 18.6 / 17.6 GiB at 4K context with r=16 (122M trainable params, 245 MB adapter); the adapter then loaded in vLLM on the w4a16 base with `status: applied` and a measurable effect vs base — [gemma-agent-lab #13](https://github.com/jwj1342/gemma-agent-lab/issues/13).
- Serving LoRA on w4a16: community adapters trained on bf16 QAT weights are served with `--enable-lora --max-lora-rank 16 --lora-modules` on the w4a16-ct base — [HF adapter card](https://huggingface.co/webAI-Official/gemma-4-12b-it-researcher-lora-peft). Marlin int4 kernels require Ampere or newer (L4 is Ada, fine) — [lushbinary guide](https://lushbinary.com/blog/gemma-4-qat-self-hosting-guide-ollama-llama-cpp-vllm/).
- Sampling defaults: temperature 1.0, top_p 0.95, top_k 64 — [HF 31B-it card](https://huggingface.co/google/gemma-4-31B-it).

### Inferences
- On L4 (Ada, 24 GB, bf16-capable) use bf16; never fp16 (Gemma lineage plus the `-1e9` fp16 overflow). L4 lacks FA3 and FA2 cannot do head_dim 512, so expect SDPA (or Unsloth's kernels); the cuDNN NaN report was H100-specific but worth a 50-step NaN check.
- Pin transformers >= the release containing PR #45312 (post Apr 9 2026) and confirm `model.config.text_config.final_logit_softcapping==30.0` is actually applied by whichever trainer you use (check the logits range on a batch).
- A full 31B bf16 replica (~62 GB) does not fit one L4; the only demonstrated 4xL4 path is naive pipeline (`device_map`) with LoRA — slow — or QLoRA (bnb 4-bit, ~22 GB), which re-quantizes differently from the served w4a16 weights. Dequantized-w4a16 -> bf16 training sees exactly the served weights; this is the HANDOFF's "serving fidelity" option and has precedent.
- Chat-template masking: build the masks yourself (Unsloth's `train_on_responses_only` markers, or render the prefix with `add_generation_prompt=True` and label only the appended model output + `<turn|>`), rather than relying on TRL's generation-marker machinery for Gemma 4.

### Gaps
- No official Google fine-tuning hyperparameters for Gemma 4; everything above is Unsloth/community.
- No measured training throughput for E4B or 12B on 4xL4 found.

## 5. Checkpoint availability

### Takeaway
Google publishes, per size, bf16 `-it`, `-it-qat-w4a16-ct` (compressed-tensors for vLLM), `-it-qat-q4_0-unquantized` (bf16 weights from the QAT pipeline), Q4_0 GGUF, and for E2B/E4B a mobile `wNa8o8` format, all Apache 2.0, all updated 2026-07-20 on HF, and (per Google docs) mirrored on Kaggle Models; the competition allows only the four `-qat-w4a16-ct` models.

### Cited Findings
| Checkpoint (HF id) | Format | Size on disk | Host | License | Source |
|---|---|---|---|---|---|
| `google/gemma-4-31B-it` | bf16, 2 shards | 62.55 GB | HF, Kaggle | Apache 2.0 | [HF API](https://huggingface.co/api/models/google/gemma-4-31B-it?blobs=true) |
| `google/gemma-4-31B-it-qat-q4_0-unquantized` | bf16 weights from QAT pipeline, 2 shards | 62.55 GB | HF | Apache 2.0 | [HF API](https://huggingface.co/api/models/google/gemma-4-31B-it-qat-q4_0-unquantized?blobs=true) |
| `google/gemma-4-31B-it-qat-w4a16-ct` | compressed-tensors W4A16 (I32+BF16 tensors) | 23.27 GB (1 file) | HF, Kaggle (competition) | Apache 2.0 | [HF API](https://huggingface.co/api/models/google/gemma-4-31B-it-qat-w4a16-ct?blobs=true) |
| `google/gemma-4-12B-it-qat-w4a16-ct` | W4A16 ct | 10.26 GB | HF, Kaggle | Apache 2.0 | [HF API](https://huggingface.co/api/models/google/gemma-4-12B-it-qat-w4a16-ct?blobs=true) |
| `google/gemma-4-E4B-it-qat-w4a16-ct` | W4A16 ct (PLE tables dominate) | 11.51 GB | HF, Kaggle | Apache 2.0 | [HF API](https://huggingface.co/api/models/google/gemma-4-E4B-it-qat-w4a16-ct?blobs=true) |
| `google/gemma-4-E2B-it-qat-w4a16-ct` | W4A16 ct | 8.32 GB | HF, Kaggle | Apache 2.0 | [HF API](https://huggingface.co/api/models/google/gemma-4-E2B-it-qat-w4a16-ct?blobs=true) |
| `google/gemma-4-31B-it-qat-q4_0-unquantized-assistant` | MTP drafter (QAT) for speculative decoding | not fetched | HF | Apache 2.0 | [HF card](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant) |

- The E4B w4a16 file (11.5 GB) is larger than the 12B one (10.3 GB): the per-layer-embedding tables (670M params x 262k vocab entries) appear to stay unquantized — inferred from sizes, consistent with the 8B total parameter count.
- Google docs: "All official Gemma 4 QAT checkpoints can also be accessed directly from Kaggle"; QAT offered for E2B, E4B, 12B, 31B; mobile format E2B/E4B only; no W4A16 for 26B-A4B (one community discussion attributes this to excessive 4-bit quality loss **[unverified]**) — [Gemma 4 overview](https://ai.google.dev/gemma/docs/core); [HF E4B w4a16 discussion](https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct/discussions/1).
- Competition: allowed models are exactly `gemma-4-31b-it-qat-w4a16-ct`, `gemma-4-12b-it-qat-w4a16-ct`, `gemma-4-e4b-it-qat-w4a16-ct`, `gemma-4-e2b-it-qat-w4a16-ct`; the harness lists their vLLM footprints as ~16-18 / 6-8 / 3-4 / 2-3 GB; PEFT dirs under `adapters/<name>/`, `max_lora_rank <= 128`, total submission < 3 GiB; vLLM TP4, `max_model_len=32768` — local `HARNESS_README.md`. Kaggle's model-list page is JS-rendered and could not be fetched; exact Kaggle Models slugs are **[unverified]** beyond the harness aliases.
- Google's recommended fine-tuning path for QAT: the QAT blog says users can "fine-tune weights directly using Hugging Face Transformers and Unsloth" (linking Unsloth's QAT page, which only mentions Unsloth Studio training with no recipe); the w4a16-ct card calls the `q4_0-unquantized` checkpoints "ideal for custom downstream compilation and research" — [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/); [HF w4a16-ct card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct).
- Memory on 4xL4 (24 GB each = 96 GB): 31B W4A16 ~19 GB weights (report: 19.2 GB Q4_0) + 1.1 GB int8 KV per 32k sequence — [arXiv 2607.02770](https://arxiv.org/abs/2607.02770) Table 3.

### Inferences
- `q4_0-unquantized` is the natural training base if dequantizing the packed checkpoint is to be avoided; it is byte-for-byte the same size as the `-it` bf16, so it would need a 62.5 GB Kaggle Dataset upload (check size limits). The cheaper route is to dequantize the 23 GB w4a16-ct inside the session (217 s per gemma-agent-lab). Whether `q4_0-unquantized` and dequantized `w4a16-ct` are numerically identical is unknown (open question).

### Gaps
- Google has not published a recommended "train-then-quantize" vs "train-on-QAT" path; nothing found beyond "fine-tune directly".
- Kaggle Models exact slugs/versions and whether the `q4_0-unquantized` variants are on Kaggle were not verifiable (JS pages).

## 6. Open questions to verify empirically in the first notebook

### Takeaway
The highest-value early checks concern template/mask fidelity and the quantized-weights training path; each is a sub-1-hour experiment.

### Cited Findings
- Context for each item is cited in sections 1-5 above.

### Inferences (checklist)
1. **Render fidelity**: with the harness's exact `chat_template.jinja` (2026-07-15 or later), render a 3-step tool trajectory with `enable_thinking=False` and `=True`, `add_generation_prompt=True/False`, and diff against the prompt vLLM actually sends (dump via vLLM request logging, as the OpenCode debuggers did). Confirm where the empty `<|channel>thought\n<channel|>` lands after tool responses and whether the base model emits it itself.
2. **Loss masking**: decode the labels after masking; ensure `<|tool_response>...<tool_response|>` and system/user turns are -100, and the `<turn|>` after each model turn is a label.
3. **Dequantized w4a16-ct vs q4_0-unquantized vs bf16 `-it`**: compare a few weight tensors (max abs diff) and eval loss; decide the training base.
4. **PEFT on the dequantized model**: confirm `get_peft_model` works after `run_compressed=False`; count targeted modules; confirm vision modules are excluded; then round-trip the adapter into vLLM on the packed w4a16-ct base (`--enable-lora --max-lora-rank r`) and check the output differs from base.
5. **Attention impl and numerics on L4**: leave `attn_implementation` unset (or `sdpa` on text only); run 50 steps in bf16 and assert no NaN; check logits are capped at +-30 (softcap applied); confirm the transformers version includes PR #45312 and that `gradient_checkpointing` no longer breaks logits.
6. **Thinking budget interaction**: with thinking on and `thinking_token_budget=4096`, measure how often the model is force-closed mid-thought on real tasks; decide between thinking-off SFT and short-thinking SFT.
7. **Tool-call robustness**: feed the vLLM `gemma4` tool parser arguments containing `}`, newlines and literal `"` (code patches) and check the round-trip; test whether the model emits parallel `<|tool_call>` blocks.
8. **12B override**: confirm the `Gemma4ForCausalLM` override loads for training as well as serving; otherwise exclude `vision_tower` by regex.
9. **Throughput**: measure tokens/s for E4B and 31B LoRA on 4xL4 at 8k and 16k sequence length (naive `device_map` pipeline vs Unsloth) to size the data budget before the Nov 12 paper deadline.
10. **Loop-behaviour baseline**: run the un-tuned w4a16-ct model on 10 tasks and count repeated identical tool calls, premature stops with tool calls as plain text, and unterminated thinking — the failure modes reported in OpenCode #21034 — to establish the metric the SFT must move.

### Gaps
- None of the above has a published answer; all require measurement.
