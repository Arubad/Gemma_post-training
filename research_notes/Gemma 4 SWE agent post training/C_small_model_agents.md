# Research cluster C: Making small (2B–8B) models reliable tool-using SWE agents

Scope: tool-call reliability, error recovery, stopping behaviour, multi-agent / multi-LoRA decomposition, context and thinking budgets, 2024–Oct 2026. Relevance judged against our setting: Gemma 4 E4B (QAT int4, vLLM TP4 on 4x L4, 32k max_model_len, LoRA rank <=128), fixed 9-tool harness, ~6 min/task, ~15 GPU-h/week/account, <$50 teacher API.

Note on sourcing: arXiv API was intermittently timing out, so abstracts were read from arxiv.org/abs pages and two full-text HTML pages (SWE-Protégé, SWE-Master, SWE-TRACE, MoRSE). Vendor/aggregator numbers are flagged as such.

## 1. Small-model SWE agent results (<=10B unless noted)

### Takeaway
Between mid-2025 and Oct 2026 the 4B–8B SWE-bench Verified frontier moved from ~15–20% (SFT on a few thousand teacher trajectories) to ~39–43% (SFT + agentic RL with loop/termination shaping, or memory/context-management training), and to ~65% for Qwen3.5-4B with a careful asynchronous RL recipe. The consistent differentiators are: (a) large, verified, diverse trajectory pools with error-step masking; (b) RL that explicitly penalises looping and rewards termination; (c) process-level (step) rewards instead of sparse outcome rewards.

### Cited Findings (SWE-bench Verified, pass@1 unless stated)

| Model / recipe | Size | SBV | What made the difference | Source |
|---|---|---|---|---|
| SWE-Gym-7B (OpenHands) | 7B | 10.6% | SFT on 491 trajectories | [SWE-Protégé Table 1](https://arxiv.org/abs/2602.22124), [SWE-Gym](https://arxiv.org/abs/2412.21139) |
| SWE-agent-LM-7B (SWE-smith) | 7B | 15.2% (paper) / 17.0% (as re-reported) | SFT on ~2.4k Claude-3.7 trajectories; scaling to 5k trajectories *regressed* to 11.8% | [SWE-smith](https://arxiv.org/abs/2504.21798); [SWE-Protégé](https://arxiv.org/abs/2602.22124) |
| Lingma SWE-GPT-7B | 7B | 18.2% | Development-process-centric synthetic data (72B: 30.2%) | [2411.00622](https://arxiv.org/abs/2411.00622) |
| SoRFT-Qwen-7B | 7B | 21.4% | Subtask decomposition (file/function/line localisation + edit) with rejection-sampled SFT + PPO | [2502.20127](https://arxiv.org/abs/2502.20127) |
| DAgger-style (Qwen3-4B / 8B) | 4B / 8B | 27.3% / 29.8% | Turn-level teacher–student mixing; base Qwen3-4B + OpenHands = 11.2%, GRPO alone 11.6% | [2605.12913](https://arxiv.org/abs/2605.12913) |
| SWE-Master-4B (Qwen3-4B-2507) | 4B | 27.6% SFT -> 33.4% RL | ~60k-instance SFT, RL with forced submission on budget exhaustion; 32B reaches 61.4% | [2602.03411](https://arxiv.org/abs/2602.03411) |
| SWE-TRACE-4B (Qwen3-4B) | 4B | 38.9% (40.7% with heuristic-guided TTS) | Cascaded "shortest-path" SFT (-21.5% tokens) + rubric process-reward RL (+2.7 over execution-only reward at 4B) | [2604.14820](https://arxiv.org/abs/2604.14820) |
| SWE-Lego-Qwen3-8B | 8B | 42.2% (49.6% TTS@16) | SFT only: 18k validated trajectories, error masking, difficulty curriculum | [2601.01426](https://arxiv.org/abs/2601.01426) |
| SWE-Protégé-7B (Qwen2.5-Coder-7B) | 7B | 42.4% with Opus-4.1 expert (~4 calls/task, 11% of tokens); 32.0% with no expert on a held-out 400-task set | SFT on 4.8k expert-augmented trajectories + RL with loop penalty; looping >10 repeated steps fell from 31% to 0.8% of trajectories | [2602.22124](https://arxiv.org/abs/2602.22124) |
| SWE-MeM-4B | 4B | 43.4% (30B: 60.2%) | Trained memory/compression tool + memory-aware GRPO with step-level credit | [2606.28434](https://arxiv.org/abs/2606.28434) |
| Arctic-SWE (Qwen3.5-4B / 9B) | 4B / 9B | 47.4 -> 65.0% / 56.2 -> 65.7% | Async RL baseline: gated reward, importance-weight control, train/infer engine alignment; ~1/6 the rollouts of FrogNano (61.5%) | [2610.24914](https://arxiv.org/abs/2610.24914) |
| Nemotron 3 Nano (30B-A3B, 3.5B active) | MoE | 38.8% (OpenHands; card does not say "Verified") | SFT with tool-calling data + GRPO across multi-step tool-use envs; BFCL v4 53.8 | [HF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| Qwen3-Coder-30B-A3B | MoE | 49.2–60.4% depending on harness (no official card number); non-thinking only | — | [search summary](https://llmrun.dev/model/qwen-qwen3-coder-30b-a3b-instruct/benchmarks) |
| Devstral Small 2 (24B dense, above our range) | 24B | 68.0% (vendor-reported) | — | [Mistral](https://mistral.ai/news/devstral-2-vibe-cli/) |
| Gemma 4 E4B (base, no SWE tuning) | ~4B eff. | ~15% (third-party aggregator, methodology unknown) | Card reports tau2-bench 42.2%, LiveCodeBench v6 52.0%, no SWE-bench | [BenchLM](https://benchlm.ai/compare/gemma-4-e4b-vs-gpt-oss-20b); [Gemma 4 card](https://ai.google.dev/gemma/docs/core/model_card_4) |

Other findings:
- Kimi-Dev: Agentless-style (localisation, edit, self-reflection) training creates "skill priors"; SFT on only 5k public trajectories then lifts SWE-Agent pass@1 to 48.6% — [2509.23045](https://arxiv.org/abs/2509.23045).
- Step Rejection Fine-Tuning: critic-judged erroneous steps kept in context but loss-masked; +3.7 vs +2.4 for plain rejection FT, reaching 32.2% — [2605.10674](https://arxiv.org/abs/2605.10674).
- Trajectory quality for LoRA SFT of Qwen2.5-Coder-7B: at 500–1000 trajectories doubling data cuts held-out CE ~12.7% while quality selection adds <1%; at 2000 trajectories quality selection gives a 3.6% gap (p=0.016); *error-retry rate* is the dominant quality sub-dimension — [2607.17205](https://arxiv.org/abs/2607.17205). Structural trajectory scoring (TQS) on 155k successful trajectories: top-ranked subsets beat random/bottom on SBV — [2610.13152](https://arxiv.org/abs/2610.13152).
- Full FT beat LoRA/QLoRA in SWE-Protégé's 7B SFT ("LoRA and QLoRA underperformed full SFT") — [2602.22124](https://arxiv.org/abs/2602.22124).

### Inferences
- A ~4B model with a fixed harness can realistically target 30–40% on SBV-like tasks with SFT+RL; the single biggest cheap lever is error-step masking and curriculum in SFT (SWE-Lego) rather than more raw trajectories (SWE-smith's 7B regression at 5k).
- Our 32k context and 150-line read cap are far tighter than any of these recipes (SWE-Master trains at 80–108k); expect the gap to the table above to come mostly from context truncation, so context-management training (SWE-MeM, Section 4) is disproportionately valuable for us.

### Gaps
- No peer-reviewed SWE-bench Verified number for Gemma 4 E4B; only a third-party aggregator (~15%).
- SWE-agent-LM-7B is reported as 15.2% in SWE-smith and 17.0% when re-reported by SWE-Protégé; unresolved.
- No evidence found on QAT-int4 effects on agentic performance of any small SWE model.

## 2. Tool-call reliability and error-recovery techniques

### Takeaway
Small models fail tool use mostly on *format adherence* and *action looping*, not on tool selection per se; the evidence favours (a) training the fixed schema in (SFT with tool data) over long schema prompts, (b) brief structured pre-call reasoning rather than long CoT, (c) loss-masked exposure to errors plus recovery steps, (d) explicit loop penalties and termination rewards in RL, and (e) harness-side budget messages and forced submission.

### Cited Findings
- SLM function-calling study: SLMs "struggle significantly with adhering to the given output format"; fine-tuning > few-shot > zero-shot — [2504.19277](https://arxiv.org/abs/2504.19277).
- Strict output-format constraints degrade reasoning; stricter formats cause larger drops — [2408.02442](https://arxiv.org/abs/2408.02442). Natural-language tool calling (decoupling tool selection from response) gives +18.4 pts accuracy and -70% output variance over JSON across 10 models, with open-weight models gaining most — [2510.14453](https://arxiv.org/abs/2510.14453). Programmatic (code) tool calling matched or beat native JSON in 11/14 models on BFCL v4 and was stable under context rot where JSON degraded 2.3% — [2608.06370](https://arxiv.org/abs/2608.06370).
- Instruction-Followed Function Calling: delegating the call-formatting step to a small dedicated model outperforms native and prompt-based function calling and stays robust under aggressive quantisation — [2608.22472](https://arxiv.org/abs/2608.22472).
- Brief is Better (Qwen2.5-1.5B, BFCL v3): 0-token CoT 44.0%, 32-token CoT 64.0%, 256-token CoT 25.0%; wrong-function selection fell from 30.5% to 1.5% with brief CoT; a fixed "Function: [name] / Key args" template matched 32-token CoT and drove function hallucination to 0.0% — [2604.02155](https://arxiv.org/abs/2604.02155).
- Looping prevalence: 33.6% (SWE-agent-LM-7B) and 24.4% (32B) of trajectories contain >10 consecutive repeated actions; after loop-penalised RL this fell to 0.8%, mean steps/task from ~60 to ~20; reward R_total = R_loop + w_follow R_follow + gates * (R_correct + ...), with hard -10 penalties in a two-phase schedule over only 160 RL steps on 100 SWE-Gym tasks — [2602.22124](https://arxiv.org/abs/2602.22124).
- SWE-smith tried harness-side action resampling when repetition is detected: repetitions fell but resolve rate slightly dropped to 38.4%; authors view repetition as a symptom of hard/OOD instances — [2504.21798](https://arxiv.org/abs/2504.21798).
- Stopping: early in SWE-Master RL ~20% of trajectories were truncated on budget exhaustion (stuck writing/modifying tests without submitting) and 24.3% of those would have passed; fix = forced submission on budget exhaustion with reward scaled by 0.5 plus per-turn "steps remaining" messages; after RL turn usage peaked at 120–140 while SFT models submitted earlier — [2602.03411](https://arxiv.org/abs/2602.03411).
- Premature submission is a catalogued hallucination mode (submitting despite visible failed assertions) — [MIRAGE-Bench 2507.21017](https://arxiv.org/abs/2507.21017). SWE-agent paper lists "edit loop" and "gave up prematurely" as failure modes — [2405.15793](https://arxiv.org/abs/2405.15793).
- Inference-time PRM (SWE-PRM) targeting redundant exploration, looping, and failure to terminate raised 40.0% -> 50.6% with taxonomy-guided feedback beating action-prescriptive feedback and shortening trajectories — [2509.02360](https://arxiv.org/abs/2509.02360). Wink: production taxonomy (spec drift, reasoning problems, tool-call failures) in ~30% of trajectories; async course-correction resolved 90% of single-intervention cases — [2602.17037](https://arxiv.org/abs/2602.17037).
- Error-recovery training: PALADIN injects failures (55+ exemplars, ToolScan taxonomy) into ToolBench to make 50k recovery-annotated trajectories; LoRA FT raised recovery rate 32.8% -> 89.7%, 95.2% retained on unseen APIs — [2509.25238](https://arxiv.org/abs/2509.25238). SWE-Lego's error masking and SRFT's masked-error-in-context both improve action quality (Section 1).
- Budget awareness: increasing tool-call budget alone fails to improve agents; a Budget Tracker that tells the agent remaining budget gives comparable accuracy at ~10x less budget — [2511.17006](https://arxiv.org/abs/2511.17006).
- Gemma 4 card claims "native function-calling support"; community reports of E4B malformed tool-call JSON and retry loops under Ollama, mitigated by low temperature and repeat penalty — [KDnuggets](https://www.kdnuggets.com/local-agentic-programming-on-the-cheap-claude-code-ollama-gemma4) (anecdotal).

### Inferences
- With only 9 fixed tools, we should bake the schema into SFT data (every trajectory uses the exact harness format) and keep the system prompt short; reserve constrained decoding (vLLM guided JSON) as a safety net only for the call envelope, not for the reasoning text, given 2408.02442.
- A trained "brief pre-call thought" (one line naming tool + key args) is the cheapest format-reliability lever and aligns with 2604.02155; long thinking before each call is counter-indicated at this scale.
- The highest-value RL reward terms for us are: repeated-action penalty, forced-submit-with-discount on budget exhaustion, and a small positive term for submitting after a passing run_command test. All are computable without a judge model.

### Gaps
- No controlled study of "schema in system prompt vs trained in" for SWE agents specifically; evidence is from function-calling benchmarks.
- No evidence on constrained decoding interacting with QAT int4 Gemma 4 in vLLM.

## 3. Decomposition and multi-LoRA evidence (for and against)

### Takeaway
Pipelines (localise -> edit -> verify) reliably beat monolithic agents when the base model is weak (Agentless, SWE-Fixer, SoRFT, MASAI), and *training* on decomposed subtasks transfers to agentic behaviour (Kimi-Dev, SoRFT). Parameter-level role specialisation via per-role LoRA on a shared base is now demonstrated (MoRSE, incl. on Gemma-4-31B) and is cheap to serve in vLLM, but no paper shows specialist adapters beating one generalist adapter *at equal data on SWE-bench*; compute-matched comparisons on reasoning favour single agents.

### Cited Findings
For decomposition:
- Agentless (localise/repair/validate, no agent loop): 32.0% SWE-bench Lite at $0.70, best open-source at the time — [2407.01489](https://arxiv.org/abs/2407.01489). MASAI modular sub-agents: 28.33% Lite, citing tuned per-sub-agent strategies and shorter trajectories — [2406.11638](https://arxiv.org/abs/2406.11638). SWE-Fixer retriever + editor (two models, two calls): 30.2% -> 32.8% SBV with filtering, trained on ~110k issues — [2501.05040](https://arxiv.org/abs/2501.05040).
- SoRFT: subtask-structured training (file/function/line/edit) gives 21.4% for a 7B — [2502.20127](https://arxiv.org/abs/2502.20127). Kimi-Dev: Agentless skills as priors make agent SFT data-efficient (5k trajectories -> 48.6%) — [2509.23045](https://arxiv.org/abs/2509.23045).
- Small critic + larger actor: 4B/8B critics (SFT+DPO, planning-level guidance only) improve six larger coding agents, e.g. +16.0 pts for GLM-4.7-Flash-30B-A3B, and cut cost ($0.07 -> $0.03 for GPT-OSS-20B) — [2606.21811](https://arxiv.org/abs/2606.21811).
- KAT-Coder-V2 "Specialize-then-Unify": five domain experts trained independently (SFT+RL) then merged by on-policy distillation into one model, 79.6% SBV; no generalist ablation reported — [2603.27703](https://arxiv.org/abs/2603.27703).
- MoRSE: (role, subtask)-conditioned LoRA experts (rank 8, last 8 blocks, 2 role + 4 subtask experts, top-2 routing) on a shared base with hierarchical GRPO; backbones Qwen3-4B, Llama-3.1-8B, Gemma-4-31B. Qwen3-4B SRDD exec 65.0 -> 86.25%, SciCode step pass 21.4 -> 29.0%; the rank-matched single fixed LoRA + GRPO baseline reached 82.5% / 25.0%, so the specialist-routing gain over a single LoRA is +3.75 / +4.0 pts (single seed); Gemma-4-31B gains were small (96.25 -> 98.65%) — [2608.09251](https://arxiv.org/abs/2608.09251).
- Subspace contention: in MoE+LoRA multi-domain tuning, near-disjoint routing still produced negative transfer because orthogonal domain gradients compete inside one low-rank adapter; separate capacity reduces it — [2609.03150](https://arxiv.org/abs/2609.03150). Hard-routed mixtures of independently trained LoRAs preserve expert behaviour with far fewer trainable parameters than soft routing — [2606.31413](https://arxiv.org/abs/2606.31413).
- Serving: vLLM selects the adapter per request via the `model` field, batches base + multiple LoRA requests together (`max_loras`), and supports runtime load/unload via `/v1/load_lora_adapter` with `VLLM_ALLOW_RUNTIME_LORA_UPDATING` — [vLLM docs](https://docs.vllm.ai/en/v0.8.1/features/lora.html). Multi-LoRA multi-agent KV sharing is an active systems topic (LRAgent, PReCache, ForkKV) — [2602.01053](https://arxiv.org/abs/2602.01053), [2609.34054](https://arxiv.org/abs/2609.34054), [2604.06370](https://arxiv.org/abs/2604.06370).

Against:
- Under equal reasoning-token budgets, single-agent systems "consistently match or outperform" multi-agent systems on multi-hop reasoning across Qwen3, R1-Distill-Llama and Gemini 2.5; MAS gains often come from unaccounted compute; MAS becomes competitive only when single-agent context utilisation degrades — [2604.02460](https://arxiv.org/abs/2604.02460).
- MAST taxonomy: 14 failure modes in 3 classes (system design, inter-agent misalignment, task verification) across 1,600+ traces — [2503.13657](https://arxiv.org/abs/2503.13657).
- SWE-smith's data-scaling regression at 7B (17.0% -> 11.8% from 2.4k to 5k trajectories) shows small models are sensitive to data mix, which splitting data across adapters would thin further — [2602.22124](https://arxiv.org/abs/2602.22124).

### Inferences
- The one condition under which 2604.02460 predicts multi-agent wins (degraded context utilisation) is exactly our regime (4B, 32k, compaction every ~15 events). A localiser sub-agent whose exploration context is discarded before the editor starts is justified on context grounds, independent of adapter specialisation.
- Multi-LoRA is operationally free in our stack (one vLLM, per-request adapter), so the real cost is splitting ~15 GPU-h/week of training. Given MoRSE's +4 pt margin over a rank-matched single LoRA at 4B (single seed) and the subspace-contention result, a 2-adapter split (localise/explore vs edit/verify) is a defensible ablation, but the generalist adapter must be the baseline.
- A tiny "critic/planner" adapter on the same base (Steer-Don't-Solve pattern) is a cheaper form of decomposition than full role agents: it only emits guidance, so it needs less data and does not touch tools.

### Gaps
- No paper compares specialist LoRA adapters vs one generalist adapter at equal SWE trajectory data; MoRSE's benchmarks are SRDD/SciCode, not SWE-bench, and single-seed.
- KAT-Coder-V2 gives no generalist-vs-specialist ablation.

## 4. Context management and thinking budget

### Takeaway
Training the agent to manage its own context (fold sub-trajectories, compress on demand) is the one small-model technique with a 4B SBV number above 40% (SWE-MeM 43.4%), and sub-trajectory isolation cuts active context ~10x without loss. On thinking: for function-calling at 1.5B the accuracy curve is non-monotonic with an optimum of 8–32 tokens; at larger scale thinking helps SWE marginally; under wall-clock budgets with fast tools, small models gain more from extra interactions than from longer thoughts.

### Cited Findings
Context:
- SWE-MeM: 4B agent trained to decide when/what/how to compress via a memory tool + memory-aware GRPO reaches 43.4% SBV (30B: 60.2%), beating static compression baselines — [2606.28434](https://arxiv.org/abs/2606.28434).
- Context-as-a-Tool (CAT): structured workspace (stable task semantics, condensed long-term memory, high-fidelity short-term) with offline trajectory-level supervision; 57.6% SBV under a bounded context budget, beating ReAct and static compression (model size not stated) — [2512.22087](https://arxiv.org/abs/2512.22087).
- Context-Folding: agent branches into a sub-trajectory, then folds it to a short summary; ~10x smaller active context, matches/beats ReAct on SWE and Deep Research, beats summarisation-based management; trained with FoldGRPO process rewards — [2510.11967](https://arxiv.org/abs/2510.11967).
- ACON: optimised compression guidelines cut peak tokens 26–54% while improving success; distilled compressors let smaller LMs gain up to 46% by reducing distraction — [2510.00615](https://arxiv.org/abs/2510.00615). Fixed-rule compression is criticised as misaligned with task needs — [ACM 2607.23809](https://arxiv.org/abs/2607.23809).
- SWE-Master trains at 80k (SFT, YaRN) and 108k (RL) context and evaluates at 128k; sequential-turn gains plateau around 125 turns — [2602.03411](https://arxiv.org/abs/2602.03411). Skywork-SWE motivates >50 rounds and >32k tokens as typical — [2506.19290](https://arxiv.org/abs/2506.19290).

Thinking budget:
- Brief is Better: at 1.5B, 32-token CoT = 64.0% vs 25.0% at 256 tokens vs 44.0% at 0; 88.6% of solvable tasks need <=32 tokens; optimum 8–16 — [2604.02155](https://arxiv.org/abs/2604.02155).
- Budget-aware tool use: raising internal thinking effort alone is suboptimal; budget tracking gives comparable accuracy at ~10x less budget — [2511.17006](https://arxiv.org/abs/2511.17006).
- Thinking vs Doing (Gemma 3 12B web agent): scaling interaction steps instead of thinking tokens, via curriculum RL on rollout length, gives SOTA open web agents — [2506.07976](https://arxiv.org/abs/2506.07976).
- Timely Machine: with wall-clock budgets, smaller models do better when tool feedback is fast and they can interact more; larger models win when tool latency is high — [2601.16486](https://arxiv.org/abs/2601.16486).
- SWE-TRACE "shortest-path" SFT reduced total tokens 21.5% at 4B and steps 10.8% while improving resolve rate; SWE-Master's thinking tokens fell ~14.8k -> 12.7k per trajectory as SFT data grew — [2604.14820](https://arxiv.org/abs/2604.14820), [2602.03411](https://arxiv.org/abs/2602.03411).
- Thinking Longer, Not Larger: a 32B with internal (multi-stage reasoning SFT) + external (search + verification) TTC reaches 46% SBV, allocating more tokens to harder problems — [2503.23803](https://arxiv.org/abs/2503.23803). Vendor data at frontier scale: Sonnet 4.5 with vs without thinking 82% vs 80% on an SBV subset — [Verdent](https://www.verdent.ai/blog/swe-bench-verified-technical-report) (vendor, small subset).
- Gemma 4 card: thinking toggled by a `<|think|>` token in the system prompt; E4B emits no empty thought block when off; no fine-grained thinking-budget control documented; E4B context 128k — [model card](https://ai.google.dev/gemma/docs/core/model_card_4).

### Inferences
- Our harness constraints (compaction every ~15 events, 150-line reads) mean the agent will see summaries it did not write; SWE-MeM/CAT suggest training the model *on trajectories that contain those compaction events* so the policy is robust to them, and Context-Folding suggests isolating exploration in a sub-agent whose output is a short structured summary.
- At 4B with a 6-minute wall clock and fast local tools, the evidence favours a small thinking budget (hundreds of tokens, or thinking off with a trained one-line pre-call rationale) and more tool interactions, not long deliberation; measure thinking_budget in {0, 256, 1024} against resolve rate and timeout rate.

### Gaps
- No study of thinking-on vs thinking-off for a ~4B model on SWE-bench specifically; evidence is from function-calling (1.5B) and web agents (12B).
- Gemma 4 model card does not document a thinking-budget parameter; our "configurable thinking_budget" may be a harness-level truncation, not a model feature.

## 5. Most promising ideas for us (ranked by expected gain per GPU-hour)

### Takeaway
Cheap data-side interventions (error masking, loop-free shortest-path trajectories, harness-format SFT) come first; short RL with loop/termination rewards second; context-management training third; multi-LoRA last and only as an ablation against a generalist adapter.

### Cited Findings (ranked)
1. **Harness-exact SFT with error masking + difficulty curriculum, 2–5k trajectories.** SWE-Lego's SFT-only 8B hit 42.2% with error masking and curriculum; SRFT shows keeping masked error steps in context beats discarding failed runs (+3.7 vs +2.4); Kimi-Dev shows 5k trajectories suffice when skills are primed — [2601.01426](https://arxiv.org/abs/2601.01426), [2605.10674](https://arxiv.org/abs/2605.10674), [2509.23045](https://arxiv.org/abs/2509.23045). Filter by error-retry rate and shortest-path, not just pass/fail — [2607.17205](https://arxiv.org/abs/2607.17205), [2604.14820](https://arxiv.org/abs/2604.14820).
2. **Short GRPO with loop penalty, forced-submit-with-discount, and "steps remaining" messages.** SWE-Protégé got looping from 31% -> 0.8% and +3.4 pts avg in 160 RL steps on 100 tasks (8 GPUs); SWE-Master's forced submission recovered the 24.3% of truncated-but-correct runs — [2602.22124](https://arxiv.org/abs/2602.22124), [2602.03411](https://arxiv.org/abs/2602.03411).
3. **Brief structured pre-call rationale ("Tool: X / args: ...") trained in, thinking budget small.** +20 pts at 1.5B, zero function hallucination with a template — [2604.02155](https://arxiv.org/abs/2604.02155); supports using Gemma's thinking only for planning turns, if at all.
4. **Compaction-aware training / exploration sub-agent with folded summary.** SWE-MeM 4B 43.4%; Context-Folding 10x smaller active context — [2606.28434](https://arxiv.org/abs/2606.28434), [2510.11967](https://arxiv.org/abs/2510.11967). Implementable in ADK as a localiser agent returning a structured "files/functions/suspect lines" summary.
5. **Synthetic tool-failure injection (PALADIN-style) on our 9 tools.** Recovery rate 32.8% -> 89.7% with LoRA; cheap because failures (bad edit_file match, 5000-char truncation, 300 s timeout) are scriptable — [2509.25238](https://arxiv.org/abs/2509.25238).
6. **Rubric/process reward from the harness (tests ran, patch applied, no repeated action) instead of outcome-only.** +2.7 pts at 4B and fewer tokens — [2604.14820](https://arxiv.org/abs/2604.14820); taxonomy-guided feedback > prescriptive — [2509.02360](https://arxiv.org/abs/2509.02360).
7. **Two-adapter split (explore/localise vs edit/verify) as an ablation vs one generalist adapter**, served by per-request LoRA in vLLM. Evidence is suggestive, not decisive: MoRSE +4 pts over a rank-matched single LoRA at 4B (single seed), subspace contention, versus compute-matched single-agent wins — [2608.09251](https://arxiv.org/abs/2608.09251), [2609.03150](https://arxiv.org/abs/2609.03150), [2604.02460](https://arxiv.org/abs/2604.02460).
8. **DAgger-style teacher correction at branch points** instead of pure teacher trajectories: +3.9 at 4B (27.3%) — [2605.12913](https://arxiv.org/abs/2605.12913); but it needs teacher calls on student states, so cap to the hardest ~10% of states to stay under $50.

### Inferences
- Items 1–3 need only SFT compute plus a handful of RL steps and are the ones to ship for the Nov 12 paper; 4–6 are the paper's novelty candidates; 7 is the headline ablation (specialist adapters vs generalist) that the field currently lacks for SWE.

### Gaps
- No published per-GPU-hour costs for these methods at 4B on L4-class hardware; rankings are inferred from reported scale (8xA100 for 160 RL steps in SWE-Protégé).

## 6. Not feasible and why

### Takeaway
Anything that relies on large-scale asynchronous RL, 100k+ contexts, or heavy frontier-model supervision is out of reach on 15 GPU-h/week, 32k context and a $50 API budget.

### Cited Findings
- Arctic-SWE's 65% at 4B depends on an asynchronous RL stack (gated reward, importance-weight control, engine alignment) and many rollouts even at 1/6 of FrogNano's; not reproducible on L4x4 — [2610.24914](https://arxiv.org/abs/2610.24914).
- SWE-Master trains at 80–108k context and evaluates at 128k; our max_model_len is 32k — [2602.03411](https://arxiv.org/abs/2602.03411).
- SWE-Protégé's headline 42.4% requires an Opus/Sonnet expert at inference (~4 calls/task, $0.13–0.15/task); without the expert it is 32.0% on a held-out subset; the hidden eval has no API access — [2602.22124](https://arxiv.org/abs/2602.22124).
- Verifier-based test-time scaling (R2E-Gym 34.4 -> 51%, SWE-Lego 42.2 -> 49.6% at TTS@16, SWE-TRACE 54 min latency per 4B run) multiplies wall-clock far beyond 6 min/task — [2504.07164](https://arxiv.org/abs/2504.07164), [2601.01426](https://arxiv.org/abs/2601.01426), [2604.14820](https://arxiv.org/abs/2604.14820).
- Full fine-tuning outperformed LoRA/QLoRA in SWE-Protégé's 7B SFT; we are limited to LoRA rank <=128 on QAT int4 — [2602.22124](https://arxiv.org/abs/2602.22124).
- Programmatic/code tool calling and natural-language tool calling (2608.06370, 2510.14453) require changing the harness's tool interface, which is fixed.
- Many-expert MoRSE-style routing (6 LoRA experts, hierarchical GRPO co-training of router + experts) is single-seed, benchmarked on SRDD/SciCode not SWE, and would split our training budget six ways — [2608.09251](https://arxiv.org/abs/2608.09251).

### Inferences
- Where a method's gain comes from TTS or an expert at inference, we can only harvest it at training time (e.g., distil verifier-selected or expert-corrected trajectories into the SFT set), not at eval.

### Gaps
- No data on LoRA-only agentic RL on a QAT int4 Gemma checkpoint; whether LoRA on quantised weights reaches the SFT quality reported for full FT is untested in the literature found.
