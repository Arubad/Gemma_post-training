# Cluster F: QAT int4 checkpoints x LoRA, and serving LoRA on compressed-tensors w4a16 in vLLM

Scope: recipe for training a LoRA adapter that vLLM serves on top of `google/gemma-4-*-it-qat-w4a16-ct`, with train-time base weights identical to serve-time base weights; memory/throughput on Kaggle 4x L4 (24 GB each, PCIe). Researched 2026-10-29.

## (1) Paper findings on QAT x LoRA and the train/serve mismatch

### Takeaway
No paper measures the exact "LoRA trained on bf16 base, served unmerged on int4 base" case; the closest controlled evidence (Aug-Sep 2026 HF forum experiment) shows Q4-vs-bf16 training paths diverge measurably (cos-sim 0.9926, ~0.146 nats mean logprob gap) while remaining task-equivalent on a narrow eval. The literature's merge-error concerns (QA-LoRA, LoTA-QAF) do not apply when vLLM keeps the adapter unmerged in bf16; the only remaining mismatch is the base weights, which "dequantize-then-LoRA" removes exactly.

### Cited Findings
- QA-LoRA (Sep 2023): base is INT4 during fine-tuning; group-wise operators let LoRA merge into the quantized model "without loss of accuracy" — motivation is merge-time error, not unmerged serving. — [arXiv 2309.14717](https://arxiv.org/abs/2309.14717)
- LoftQ (Oct 2023): there is "a consistent gap" between full FT and quantization+LoRA; LoftQ finds a low-rank init that "alleviates the discrepancy between the quantized and full-precision model", strongest at 2-bit. PEFT implements it only for bitsandbytes nf4 (`replace_lora_weights_loftq`, one iteration). — [arXiv 2310.08659](https://arxiv.org/abs/2310.08659); [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)
- LQ-LoRA (Nov 2023): decomposes each matrix into a quantized component plus a high-precision low-rank component that is trained. — [arXiv 2311.12023](https://arxiv.org/abs/2311.12023)
- L4Q (Feb 2024, ACL 2025): joint quantization + LoRA; criticizes the two-stage "PTQ (e.g. GPTQ) then LoRA on frozen quantized weights" pipeline. — [arXiv 2402.04902](https://arxiv.org/abs/2402.04902)
- LoTA-QAF (May 2025): names three problems of LoRA on quantized models: dtype mismatch int4 vs 16-bit adapter; "accuracy degradation when merging" the adapter into int4; no lossless merge. Important caveat: "For task-specific fine-tuning, 16-bit LoRA achieves superior results" over their ternary method (i.e., keeping a bf16 adapter unmerged is the accuracy-max path). — [arXiv 2505.18724](https://arxiv.org/abs/2505.18724)
- CoA-LoRA (Sep 2025): a LoRA that adapts to arbitrary per-layer bit-width configurations without retraining; confirms that a LoRA is configuration-specific in general. — [arXiv 2509.25214](https://arxiv.org/abs/2509.25214)
- RoSTE (Feb 2025) and QWHA (Sep 2025) are additional quantization-aware SFT/PEFT methods; none address Gemma QAT checkpoints. — [arXiv 2502.09003](https://arxiv.org/abs/2502.09003); [arXiv 2509.17428](https://arxiv.org/abs/2509.17428)
- Controlled mismatch experiment (Qwen3-4B tool-use LoRA trained against bf16, Q8_0, Q4_K_M copies; posted 22 Aug 2026): bf16-vs-Q8 adapters cos-sim 0.9998, bf16-vs-Q4 0.9926; teacher-forced mean |delta logprob| 0.020 nats (bf16 vs Q8) vs 0.146 nats (bf16 vs Q4); 3/36 boundary prompts flipped tool choice for Q4; all three still scored 1.00 on 29 held-out prompts. Author: training through deployed quantization and through the bf16 parent "are not numerically the same operation". — [HF forum](https://discuss.huggingface.co/t/deterministic-lora-training-directly-through-quantized-gguf-serving-weights/179118)
- Recover-LoRA (Edge0-35B-A3B, Sep 2026, vendor blog): LoRA trained by distillation on a frozen int4 base with fp16 teacher; final int4+LoRA averages 79.2 vs fp16 83.2 (3.9-point gap; HumanEval 90.9 vs 95.1). No PTQ-only baseline given. — [MindStudio blog](https://www.mindstudio.ai/blog/edge0-35b-quantization-quality-loss/)
- Google: QAT lets models "perform nearly identically to their high-precision baselines"; the 31B card says QAT preserves "similar quality to bfloat16". No numbers for the w4a16 checkpoint vs bf16. — [Gemma docs](https://ai.google.dev/gemma/docs/core); [31B w4a16 card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)
- torchtune QAT docs: the same quantizer used during QAT fine-tuning must be used when quantizing after, "or the numerics will be off". — [torchtune](https://docs.pytorch.org/torchtune/0.6/tutorials/qat_finetune.html)
- Unsloth QAT (torchao `qat_scheme` in `int4`, `fp8-int4`, etc.): fake-quantizes frozen base and LoRA during training, exports int4 via `save_pretrained_torchao`; claims ~2.1% raw accuracy gain / up to 66.9% recovery vs PTQ (vendor-reported). — [Unsloth QAT](https://unsloth.ai/docs/blog/quantization-aware-training-qat); [torchao QAT](https://docs.pytorch.org/ao/main/workflows/qat.html)

### Inferences
- vLLM serves LoRA as base-GEMM (int4 Marlin) + separate bf16 LoRA GEMM, never merged, so the merge-error line of work (QA-LoRA, LoTA-QAF) is irrelevant here; LoTA-QAF's own finding that unmerged 16-bit LoRA wins on task-specific FT supports the plan.
- The only mismatch risk is the frozen base. Dequantizing the w4a16 checkpoint to bf16 gives weights bit-identical to what Marlin reconstructs (int4 x bf16 scale), so train = serve up to kernel accumulation order. Training on `-qat-q4_0-unquantized` (the QAT bf16 parent) would be the "0.146 nats" class of mismatch at worst, likely smaller because QAT weights already sit near the grid; training on a bnb-nf4 requantization adds a second, different quantization error and is the worst option.
- Unsloth QAT/torchao export produces a torchao int4 checkpoint, not Google's compressed-tensors base, so it cannot be used when the serve-time base is fixed to `-qat-w4a16-ct`.

### Gaps
- No paper or benchmark measures bf16-trained LoRA served unmerged on an int4 base at SWE-agent scale; the HF forum experiment is one model, 29-36 prompts.
- No public accuracy table for Gemma 4 `-qat-w4a16-ct` vs bf16 (Google gives only qualitative claims).

## (2) The compressed-tensors w4a16 format and the dequantize-then-LoRA recipe

### Takeaway
All three target checkpoints are `pack-quantized`, int4 symmetric, group size 32, Linear-only, activations unquantized, with `lm_head` and the vision/audio towers (and for E4B the audio tower) in the `ignore` list; `transformers` `CompressedTensorsConfig(dequantize=True)` or `compressed_tensors.entrypoints.convert.convert_checkpoint(..., CompressedTensorsDequantizer)` yields a plain bf16 model whose Linear weights equal the served int4 values.

### Cited Findings
- 31B config: format `pack-quantized`, quant_method compressed-tensors v0.15.1.a20260521, group_0 targets `Linear`, weights 4-bit int, symmetric, strategy group, group_size 32, observer memoryless_minmax, actorder null, input/output activations null, kv_cache_scheme null. Ignore (192): `lm_head`, vision patch embedder input proj, vision embedding projection, all 7 linears in each of 27 vision layers. Text: hidden 5376, 60 layers (10 full/50 sliding), intermediate 21504, 32 heads, head_dim 256 (global 512), 16 sliding KV heads / 4 global, `attention_k_eq_v: true`, vocab 262144, tied embeddings, max_position 262144. Safetensors: 33B params, dtypes I32 + BF16. — [31B config.json](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct/raw/main/config.json); [card](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)
- E4B config: same scheme (int4 sym g32). Ignore (250): `lm_head`, 134 audio-tower modules, 113 vision-tower modules, `embed_vision/embed_audio.embedding_projection`. NOT in ignore (so they are not Linear or are PLE lookups): `embed_tokens_per_layer`, `per_layer_model_projection` (no entry). Text: hidden 2560, 42 layers (35 sliding/7 full), intermediate 10240, 8 heads, 2 KV heads, head_dim 256, `hidden_size_per_layer_input` 256, `vocab_size_per_layer_input` 262144, `num_kv_shared_layers` 18, sliding_window 512, max_position 131072. Safetensors ~9B params. — [E4B config.json](https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct/raw/main/config.json); [E4B card](https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct)
- 12B config: arch `Gemma4UnifiedForConditionalGeneration`, same int4 sym g32; ignore only 17 entries (vision/audio embedding projections, `lm_head`). Text: hidden 3840, 48 layers (8 full/40 sliding), intermediate 15360, 16 heads, 8 KV (1 global), head_dim 256 (global 512), PLE off (`hidden_size_per_layer_input` 0), no KV sharing, max_position 262144. — [12B config.json](https://huggingface.co/google/gemma-4-12B-it-qat-w4a16-ct/raw/main/config.json)
- transformers loading modes: default = weights left compressed and decompressed on first forward; `CompressedTensorsConfig(dequantize=True)` = "dequantized to the model dtype (e.g. BF16)... the model can be fine-tuned or saved in that dtype"; `use_optimized_inference=True` is FP8-only, inference-only. Doc states "compressed-tensors supports fine-tuning with PEFT". Code: `AutoModelForCausalLM.from_pretrained(id, quantization_config=CompressedTensorsConfig(dequantize=True), device_map="auto")`. — [transformers compressed-tensors docs](https://huggingface.co/docs/transformers/main/en/quantization/compressed_tensors)
- Offline conversion: `from compressed_tensors.entrypoints.convert import convert_checkpoint, CompressedTensorsDequantizer`; `convert_checkpoint(model_stub=..., save_directory=..., converter=CompressedTensorsDequantizer(model_stub, ignore=[...], dtype=torch.bfloat16))`; handles W4A16/INT8, reads `quantization_config`, removes it from the output config. Example given is the W4A16 Kimi-K2.6 checkpoint. — [llm-compressor convert docs](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/convert/)
- Per-module primitive used by the transformers integration: `compressor.decompress_module(module)`. — [transformers integration source (mirror)](https://download.hermesedi.duckdns.org/venv_trading/lib/python3.11/site-packages/transformers/integrations/compressed_tensors.py)
- Google bf16 weight sizes: E4B 17.9 GB, 12B 26.7 GB, 31B 69.9 GB (Q4_0: 4.5 / 6.7 / 17.5 GB). — [Gemma docs](https://ai.google.dev/gemma/docs/core)

### Inferences
- Effective bits for g32 int4 with bf16 scales = 4.5 bits/weight; on-disk w4a16-ct estimates (int4 linears + bf16 everything else): E4B ~10.5 GB (PLE table alone 262144x256x42x2 B = 5.6 GB bf16), 12B ~8 GB, 31B ~17-18 GB. Third-party blogs quote 10.3 GB (12B) and ~18 GB (31B), unverified.
- Dequantized bf16 for training equals Google's bf16 numbers: E4B 17.9 GB, 12B 26.7 GB, 31B 69.9 GB. E4B text-only training can drop vision+audio towers (~1-1.5 GB) and keep the frozen PLE table on CPU, bringing GPU-resident weights to roughly 9 GB.
- Recommended one-time step (needs internet, so do it in a CPU/internet-enabled session or locally): run `convert_checkpoint` with `CompressedTensorsDequantizer`, save bf16 safetensors, upload as a private Kaggle Dataset, and train from that. Sanity check: `unpack`/dequant one layer and compare to the first-forward decompressed weight from the default loading path (should be exactly equal).

### Gaps
- Not verified whether `dequantize=True` leaves `model.config.quantization_config` / `hf_quantizer` state that makes `Trainer`/`save_pretrained` treat the model as quantized; convert-then-reload avoids this.
- Exact w4a16-ct repo sizes on disk were not visible in the fetched cards.

## (3) Tooling compatibility matrix (transformers / PEFT / TRL / Unsloth / vLLM x Gemma 4 x compressed-tensors x LoRA)

### Takeaway
vLLM marks `Gemma4ForCausalLM` and `Gemma4ForConditionalGeneration` (E2B/E4B/31B) as LoRA-capable and lists compressed-tensors w4a16 as LoRA-tested, but `Gemma4UnifiedForConditionalGeneration` (12B) has NO LoRA mark in the supported-models table; PEFT/TRL work on the dequantized bf16 model; Unsloth supports Gemma 4 but is undocumented for compressed-tensors inputs.

### Cited Findings
| Component | Status | Source (date) |
|---|---|---|
| vLLM LoRA on Gemma 4 E-series/31B | `Gemma4ForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsQuant, SupportsPP, SupportsLoRA, ...)`; `packed_modules_mapping` qkv_proj->(q,k,v), gate_up_proj->(gate,up); `supports_tower_connector_lora=True`; no `embedding_modules` defined (so no embed_tokens/lm_head LoRA). | [vLLM gemma4_mm.py](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/gemma4_mm.py) |
| vLLM LoRA on Gemma 4 12B (Unified) | Supported-models table: `Gemma4UnifiedForConditionalGeneration` LoRA column blank; PP yes. | [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html) |
| vLLM LoRA on compressed-tensors base | Tracker lists compressed_tensors w4a16 (qwen3-30b-a3b-w4a16) as working, alongside GPTQ/AWQ/FP8/MXFP4; bitsandbytes and GGUF not supported. Issue closed "not planned" May 2026. | [vLLM #29424](https://github.com/vllm-project/vllm/issues/29424) |
| vLLM Gemma 4 weight-loading rules | KV-shared layers: `k_proj`, `v_proj`, `k_norm` dropped at load; full-attention layers with `attention_k_eq_v`: K loaded into K and V shards, "no v_proj is expected"; tower quant check: "BNB/torchao/compressed-tensors handle arbitrary sizes; other methods (Marlin, FP8) require dimensions divisible by 64". | [vLLM gemma4 API doc](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/gemma4/); gemma4_mm.py above |
| vLLM `--max-lora-rank` | Default 16; allowed `Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]`. | [vLLM config/lora](https://docs.vllm.ai/en/v0.13.0/api/vllm/config/lora/) |
| vLLM adapter format | Directory containing `adapter_config.json` (HF resolver requires it); `--lora-target-modules` takes suffixes (`o_proj`, `qkv_proj`, `gate_proj`, `down_proj`); multimodal tower/connector LoRA is experimental. | [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora.html) |
| vLLM LoRA + embeddings on quantized base | Community report: GPTQ base + adapter containing `embed_tokens` fails with "unsupported LoRA weight"; workaround strip those tensors. | [vLLM forum](https://discuss.vllm.ai/t/support-for-deploying-4-bit-fine-tuned-model-with-lora-on-vllm/1186) |
| transformers x compressed-tensors | Loads pre-quantized checkpoints; `dequantize=True` for fine-tuning; "supports fine-tuning with PEFT". | [transformers docs](https://huggingface.co/docs/transformers/main/en/quantization/compressed_tensors) |
| PEFT quantization guide | Lists bitsandbytes, GPTQModel, AQLM, EETQ, HQQ, torchao (int8 only; `int4_weight_only` not supported), INC, Transformer Engine; does not mention compressed-tensors. | [PEFT guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization) |
| Axolotl x Gemma 4 | Must use regex targets `model.language_model.layers.[\d]+.(_checkpoint_wrapped_module.)?(mlp\|self_attn).(up\|down\|gate\|q\|k\|v\|o)_proj`, never `lora_target_linear: true` (hits vision/audio, loss ~12). `use_reentrant: false` required. FSDP2 works with `transformer_layer_cls_to_wrap: Gemma4TextDecoderLayer`; FSDP2 + activation offloading ~26 GiB/GPU for 26B-A4B. PEFT `Gemma4ClippableLinear` error auto-patched. | [Axolotl Gemma 4 notes](https://docs.axolotl.ai/docs/agents/model_architectures.html) |
| Unsloth x Gemma 4 | Supports E2B/E4B/26B-A4B/31B (12B mentioned once); QLoRA 31B ~22 GB, E4B LoRA 10-17 GB; Gemma 4 RL with vLLM not supported (`fast_inference=False`); Gemma 4 fixes Apr 8 2026; multi-GPU, QAT-ct inputs not mentioned. | [Unsloth Gemma 4 guide](https://unsloth.ai/docs/models/gemma-4/train); [changelog](https://unsloth.ai/docs/new/changelog) |
| Third-party version note | Community W4A16 Gemma 4 build says transformers >= 5.10.1 needed; vLLM <= 0.27.1 for transformers >= 5.15 (unverified pairing). | [Friendli model page](https://friendli.ai/models/GotoAI-Inc/gemma-4-31B-it-W4A16) |
| Checkpoint availability | HF collection "Gemma 4 QAT Q4_0" (updated Jul 21 2026): `google/gemma-4-{E2B,E4B,12B,31B}-it-qat-w4a16-ct` (6B/9B/13B/33B params), `*-qat-q4_0-unquantized` for all 5 sizes incl. 26B-A4B, `*-qat-q4_0-gguf`, `*-assistant` MTP drafters. License Apache 2.0 (per model cards). Google docs say QAT checkpoints are on Kaggle at `kaggle.com/models/google/gemma-4/transformers`. QAT release announced ~June 5 2026. | [HF collection](https://huggingface.co/collections/google/gemma-4-qat-q4-0); [Google blog](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/); [Gemma docs](https://ai.google.dev/gemma/docs/core) |

### Inferences
- The 12B is the riskiest serving target: its vLLM class lacks a LoRA mark. If the competition's serving config is fixed to vLLM + LoRA, 12B may be unusable until verified against the exact vLLM version in the submission environment.
- For 31B, the HF model has no `v_proj` on `attention_k_eq_v` full-attention layers (vLLM expects none); for E4B, 18 KV-shared layers have their `k_proj/v_proj` dropped at vLLM load. A PEFT adapter that contains LoRA tensors for those modules would either be rejected or silently ignored; restrict `target_modules` by regex to language-model layers and check `adapter_model.safetensors` keys against vLLM's expected names.
- Do not target `embed_tokens`, `lm_head`, `embed_tokens_per_layer` or `per_layer_model_projection`: no `embedding_modules` in vLLM's Gemma 4 class, and lm_head is tied.

### Gaps
- Kaggle Models page is JS-rendered and could not be read; the exact Kaggle variant names/sizes for `-qat-w4a16-ct` are unverified (Google docs assert presence).
- No source confirms Unsloth can load a compressed-tensors checkpoint (its QLoRA path uses bitsandbytes).
- No source confirms that vLLM's LoRA path handles the 31B `k_eq_v` layers' qkv_proj packing when an adapter supplies `k_proj` only.

## (4) Memory/throughput estimates on 4x L4 (24 GB, 121 TFLOPS dense bf16, 300 GB/s, PCIe Gen4 x16 64 GB/s)

### Takeaway
E4B bf16 LoRA at 16-32k fits per-GPU with text-only loading + fused CE; 12B bf16 LoRA needs FSDP2 sharding across the 4 GPUs (26.7 GB weights); 31B is infeasible in bf16 (69.9 GB) and only feasible with a 4-bit-resident base sharded by FSDP at <=16k, at roughly 0.4-0.5k tok/s.

### Cited Findings
- L4: BF16 tensor 121 TFLOPS dense (242 sparse), 24 GB, 300 GB/s, PCIe Gen4 x16 64 GB/s, 72 W. — [NVIDIA L4](https://www.nvidia.com/en-us/data-center/l4/)
- bf16 weights: E4B 17.9 GB, 12B 26.7 GB, 31B 69.9 GB. — [Gemma docs](https://ai.google.dev/gemma/docs/core)
- Vocab 262144 for all sizes; hidden 2560/3840/5376; layers 42/48/60; intermediate 10240/15360/21504. — configs cited in (2)
- FSDP-QLoRA (bitsandbytes + FSDP) trains up to 70B on dual 24 GB GPUs; PEFT ships `fsdp_config_qlora.yaml`/`run_peft_qlora_fsdp.sh`. — [bitsandbytes FSDP-QLoRA](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora); [AnswerDotAI fsdp_qlora](https://awesome.ecosyste.ms/projects/github.com%2Fanswerdotai%2Ffsdp_qlora)
- Liger `FusedLinearCrossEntropy` never materializes full logits; README claims ~60% memory reduction, 20% throughput gain; Gemma kernel listing includes FusedLinearCrossEntropy in recent versions. — [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)
- Axolotl: FSDP2 works for Gemma 4; 26B-A4B at ~26 GiB/GPU with activation offloading. — [Axolotl](https://docs.axolotl.ai/docs/agents/model_architectures.html)
- Unsloth: 31B QLoRA ~22 GB (single GPU, short seq); E4B LoRA 10-17 GB. — [Unsloth](https://unsloth.ai/docs/models/gemma-4/train)

### Inferences (estimates; 8N FLOPs/token with checkpoint recompute, ~35% MFU = ~42 TFLOPS/GPU)
| | E4B (text-only ~4.5B incl. lm_head) | 12B (~11.8B) | 31B (~30B) |
|---|---|---|---|
| Logits at 32k, bf16, unfused | 17 GB (must use fused/chunked CE at any size) | 17 GB | 17 GB |
| Per-layer hidden checkpoints, 16k / 32k | 0.084x42 = 3.5 / 7 GB | 0.126x48 = 6 / 12 GB | 0.176x60 = 10.5 / 21 GB |
| Recommended base layout | bf16 dequantized, towers dropped, PLE table on CPU: ~9 GB/GPU, DDP x4 | bf16 dequantized, FSDP2 x4: ~6.7 GB/GPU weights | int4-resident (16-18 GB total) sharded FSDP x4: ~4.5 GB/GPU |
| Fits 16k / 32k? | yes / yes (tight at 32k, ~9+7+3 GB) | yes / tight (6.7+12+3+recompute ~22 GB; use activation offload or 24k) | 16k only (~4.5+10.5+3 = ~18-19 GB); 32k needs CPU activation offload |
| tok/s per GPU | ~1,100 | ~450 | ~170 (before dequant overhead) |
| tok/s on 4 GPUs | ~2,500-4,500 (DDP) | ~1,200-1,600 (FSDP, PCIe gathers ~5-10%) | ~350-550 |
| Tokens per 12 h session | ~100-190M | ~50-70M | ~15-24M |
| 40M-token epoch (2k traj x 20k) | ~3-4 h | ~8-10 h (one session) | ~20-30 h (3 sessions, all weekly quota of 2 accounts) |
| LoRA r=64 all-7-linears size (bf16) | ~0.29 GB | ~0.52 GB | ~0.97 GB (r=128 ~1.9 GB) |

- PCIe all-gather cost for FSDP 12B: ~16 GB received per GPU per gather x3 per step (fwd, recompute, bwd) at ~20-30 GB/s effective = 1.5-2.5 s vs ~20 s compute per 32k-token step; acceptable.
- The 31B int4-resident path has no off-the-shelf exact-match implementation: bnb nf4 is a different grid (requantization error); torchao int4 is uint4+zero-point and PEFT does not support LoRA on it; HQQ supports LoRA but uses its own quantizer. An exact-match "frozen int4-g32 linear" module (store packed int32 + bf16 scales, dequantize per forward with compressed-tensors' unpack/dequant, apply LoRA) is ~50-100 lines but is unvalidated engineering.
- Multi-GPU inside Kaggle notebooks: `accelerate launch` / `torchrun` from a notebook cell works in principle (subprocess), but `notebook_launcher` with FSDP has had OOM/AttributeError reports; DeepSpeed ZeRO-3 is not documented for Gemma 4 anywhere found, so prefer FSDP2 (Axolotl-verified).

### Gaps
- No Kaggle-specific L4x4 FSDP/QLoRA notebook or quota documentation was found; sources only describe T4/P100 30 h/week (the user's 15 GPU-h/week figure is not independently confirmed).
- All throughput numbers are derived, not measured; no public Gemma 4 LoRA tok/s on L4 was found.
- Kaggle Dataset size limits for a 70 GB bf16 31B conversion were not checked.

## (5) Recommended recipe and fallbacks

### Takeaway
Train LoRA with PEFT/TRL on the bf16 dequantization of the exact `-qat-w4a16-ct` checkpoint, restricted by regex to language-model q/k/v/o/gate/up/down, r<=128 (set `--max-lora-rank` accordingly), export a plain PEFT directory, and serve it unmerged in vLLM on the same `-qat-w4a16-ct` base with TP=4; E4B first, 12B only after vLLM LoRA support is confirmed, 31B only if the int4-resident trainer is proven in a smoke test.

### Cited Findings
- vLLM applies LoRA to all supported modules when `--lora-target-modules` is unset; adapter dir must contain `adapter_config.json`. — [vLLM LoRA docs](https://docs.vllm.ai/en/latest/features/lora.html)
- Unsloth recommended SFT hyperparameters for Gemma 4: r=8, alpha=8, dropout 0, lr 2e-4 (2e-5 for long runs), adamw_8bit, warmup 5, max_seq 8192; chat template `gemma-4-thinking` for 26B/31B, `gemma-4` for small; mismatched chat template/EOS is "the most common cause" of worse behavior in other runtimes. — [Unsloth](https://unsloth.ai/docs/models/gemma-4/train)
- Axolotl regex target modules and `use_reentrant: false`; `Gemma4TextDecoderLayer` as FSDP2 wrap class. — [Axolotl](https://docs.axolotl.ai/docs/agents/model_architectures.html)
- `CompressedTensorsDequantizer` / `dequantize=True` APIs. — [llm-compressor convert](https://docs.vllm.ai/projects/llm-compressor/en/latest/guides/entrypoints/convert/); [transformers](https://huggingface.co/docs/transformers/main/en/quantization/compressed_tensors)

### Inferences
1. Primary (E4B, lowest risk): offline `convert_checkpoint(... CompressedTensorsDequantizer ...)` -> bf16 safetensors -> Kaggle Dataset. In the GPU notebook: load bf16, `freeze` towers (or load text-only class), keep PLE table frozen (CPU if memory-bound), PEFT `LoraConfig(r=64, alpha=64, dropout=0, target_modules=r"...language_model.layers.\d+.(self_attn|mlp).(q|k|v|o|gate|up|down)_proj")`, gradient checkpointing non-reentrant, Liger fused linear CE, packing with 16k-32k max length, DDP x4 (or FSDP2 if 32k OOMs). Save with `peft_model.save_pretrained()` -> `adapter_config.json` + `adapter_model.safetensors`. Strip any tensor whose module vLLM drops (k/v on KV-shared layers) before submission.
2. 12B: same pipeline with FSDP2 (`Gemma4TextDecoderLayer` wrap) across 4 GPUs. Gate on a vLLM smoke test: `vllm serve google/gemma-4-12B-it-qat-w4a16-ct --enable-lora --max-lora-rank 64 --tensor-parallel-size 4 --max-model-len 32768` with a random-init adapter; if the Unified class rejects LoRA, drop 12B.
3. 31B: only via a 4-bit-resident base sharded by FSDP at <=16k; prefer an exact int4-g32 dequant module over bnb nf4 (which trains against different weights). Budget 3+ sessions per epoch; use TRL checkpoint resume.
4. Fallback if exact-match training is impossible for a size: train on `-qat-q4_0-unquantized` bf16 weights (same QAT parent, same int4 grid family as Q4_0); expected mismatch is in the "0.02-0.15 nats" class, smaller than for a non-QAT model. Last resort: bnb-nf4 QLoRA on the dequantized weights (double quantization error; validate with the vLLM eval harness, not training loss).
5. Serve: `--enable-lora --max-lora-rank <r> --max-loras 1 --tensor-parallel-size 4 --max-model-len 32768`; keep `--lora-target-modules` unset; one adapter per request.

### Gaps
- No end-to-end public example of PEFT training on a dequantized compressed-tensors Gemma 4 checkpoint and serving it with vLLM LoRA; the recipe is composed from documented parts.

## (6) Open risks to verify in the first Kaggle notebook

### Takeaway
Six checks decide feasibility: vLLM LoRA on the exact checkpoint/version, 12B Unified LoRA support, adapter key compatibility with Gemma 4's shared/eq-KV layers, bit-equality of dequantized vs served weights, 32k memory with fused CE, and Kaggle multi-GPU launch.

### Cited Findings
- 12B `Gemma4UnifiedForConditionalGeneration` has no LoRA mark in vLLM's table. — [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- vLLM drops `k_proj/v_proj/k_norm` on KV-shared layers and expects no `v_proj` on `attention_k_eq_v` layers. — [vLLM gemma4 API](https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/gemma4/)
- Embedding tensors in adapters fail on quantized bases in vLLM ("unsupported LoRA weight"). — [vLLM forum](https://discuss.vllm.ai/t/support-for-deploying-4-bit-fine-tuned-model-with-lora-on-vllm/1186)
- Gemma 4 LoRA starts at loss ~12 when vision/audio linears are targeted; E2B/E4B start high even with correct targets (PLE suspected). — [Axolotl](https://docs.axolotl.ai/docs/agents/model_architectures.html)
- Unsloth: Gemma 4 + vLLM fast inference unsupported in RL; `use_cache=False` garbage-logit bug fixed; fp16 audio overflow on T4-class. — [Unsloth](https://unsloth.ai/docs/models/gemma-4/train)
- vLLM LoRA-on-quantized tracker closed "not planned" with an unresolved AWQ garbage-output report (Jan 2026). — [vLLM #29424](https://github.com/vllm-project/vllm/issues/29424)

### Inferences (checklist)
1. Pin and record the competition vLLM/transformers/compressed-tensors versions; run `vllm serve <ct checkpoint> --enable-lora --max-lora-rank 64 -tp 4` with a zero-init adapter and assert outputs equal the base model's (catches kernel/LoRA path bugs like the AWQ report).
2. Compare `adapter_model.safetensors` keys to vLLM's expected module names for Gemma 4 (qkv_proj packing, KV-shared layers, k_eq_v layers); strip or confirm.
3. Assert `torch.equal(dequantized_weight, first_forward_decompressed_weight)` for a few layers; also compare to vLLM's loaded Marlin-dequantized values if feasible.
4. Measure peak memory for one 32k packed sequence with Liger fused CE on each size; fall back to 16k/24k or activation offload.
5. Confirm `torchrun`/`accelerate launch` FSDP2 from a Kaggle cell on 4x L4 and that `Gemma4TextDecoderLayer` wrapping works with PEFT.
6. Check starting loss (~1-3 expected for correct text-only targeting on 12B/31B; E4B may start high per Axolotl) and that the chat template/EOS used in training equals vLLM's.
7. Verify Kaggle Models actually hosts `-qat-w4a16-ct` (unreadable here) or attach the HF files as a Kaggle Dataset; total submission <3 GiB rules out r>128 on 31B only if other files are large.

### Gaps
- Whether the competition's vLLM build enables `--enable-lora` with TP=4 and compressed-tensors Marlin on L4 (SM89) is untested here.
