# Research cluster E: Test-time compute scaling and inference-time strategies for SWE agents under a hard wall-clock budget

Scope: 2024–Oct 2026 literature on sampling, verification, reranking, search and budget control for repo-level issue resolution, judged against our setting (Gemma 4 E4B/12B/31B int4 + LoRA, vLLM TP4 on 4x L4, ~120 Python tasks in 12 h total, ADK YAML agents, 9 fixed tools, thinking_budget 0–32768, hidden tests; our harness strips test-file edits from the final patch).

## (1) Method table: what each TTS strategy buys and what it costs

### Takeaway
Every published TTS gain for SWE agents comes from running 8–40 full trajectories per task and selecting with execution and/or a learned verifier; typical gains are +6 to +17 points absolute at 8–40x token cost. The only methods that improve the latency/accuracy frontier rather than just the ceiling are step-level pruning (SWE-TRACE), trajectory replay (SWE-Replay), early termination (EET/FailFast/StopLoss) and cheap independent review (Rethinking Verification).

### Cited Findings
| arXiv ID | Strategy | N | Verifier | Gain (SWE-bench Verified unless noted) | Cost |
|---|---|---|---|---|---|
| 2407.01489 Agentless | Pipeline: 40 patches (4 location sets x 10), 40 reproduction tests, LLM-filtered regression tests, then majority vote on AST-normalized patches | 40 patches, 40 tests | Execution (repo regression tests + generated repro tests) + majority vote | Lite: majority-vote-only 25.67% -> +regression 27.00% -> +repro tests 32.00%; Verified 38.80% (GPT-4o) | $0.70/issue on Lite (~78k tokens) — [Source](https://arxiv.org/html/2407.01489) |
| 2501.14723 CodeMonkeys | 10 parallel testing/editing state machines x up to 8 serial iterations; selection by model-written tests (top-3) then a selection state machine | 10 | Execution on model-written tests + LM selection | Coverage (oracle) 69.8%, random pick 45.8%, selected 57.4% (recovers ~half of oracle gap); ensemble "Barrel of Monkeys" 66.2% | ~$2,292 total for 500 tasks with Claude 3.5 Sonnet; edit generation 59.6% of cost. "Early serial iterations matter most"; many parallel candidates "can make selection more difficult" — [Source](https://arxiv.org/html/2501.14723) |
| 2410.20285 SWE-Search | MCTS + Value Agent + Discriminator debate (Moatless scaffold) | search tree | LM value function | "23% relative improvement" over non-MCTS agents across five models | not reported — [Source](https://arxiv.org/abs/2410.20285) |
| 2509.23045 Kimi-Dev | BugFixer x TestWriter self-play: 40 patches x 40 tests, test-pass matrix (fail-to-pass + pass-to-pass score) | 40x40 | Execution on self-generated tests, filtered to those that fail on the original repo | 48.0% (1x1) -> 60.4% (40x40); a 3x3 self-play already beats majority voting over 40 patches; 72B model. SWE-Agent SFT variant: 48.6% pass@1, 74.0% pass@10 | 40 patch samples + 40 tests + 1600 test runs per issue — [Source](https://arxiv.org/html/2509.23045) |
| 2506.19290 Skywork-SWE | Best-of-N trajectories reranked by OpenHands critic | 8 | LM critic | 38.0% -> 47.0% (32B) | 8x rollouts — [Source](https://arxiv.org/abs/2506.19290) |
| 2505.13652 Nebius guided search | Trained action-value critic for (a) 1-step lookahead (K=4 candidate actions/step) and (b) best-of-N trajectory selection | N=5–15, K=4–8 | Trained critic initialized from LLaMA-3.1-70B on ~80k pos/neg trajectories | Qwen2.5-72B policy 16.2% (default) / 22.8% (until-submitted) -> selection N=10 31.3%/37.6%; lookahead+selection N=10 41.7%/44.1%; full Verified K=8,N=15: 40.8%. GPT-4o 22.0% -> 40.0% (N=5, lookahead+selection) | N full trajectories + K action samples per step; "until submitted" adds ~1.65 extra attempts/problem — [Source](https://arxiv.org/html/2505.13652) |
| 2504.07164 R2E-Gym hybrid verifiers | Best@K over 32B agent rollouts | K=16–26 | Execution-based (test-writing agent, M=10 tests, regression-score gating) saturates ~43.7%; execution-free 14B LoRA verifier saturates ~42.8%; hybrid = Top_n(EF)+EB | 34.4% pass@1 -> Best@16 49.4% -> Best@26 51.0%. Regression tests alone 47.4%, Agentless-style tests 48.8%, trained test agent 51.0%. Adding 5 test-agent rollouts (49.3%) beats adding 5 more editing rollouts (47.6 -> 48.4%) | 16–26 editing rollouts + 5–10 test rollouts; test rollouts "considerably cheaper" — [Source](https://arxiv.org/html/2504.07164) |
| DeepSWE-Preview (Agentica, HF card) | Same hybrid TTS on RL-trained 32B | 8, 16 | Hybrid (EB + 14B DeepSWE-Verifier) | 42.2% pass@1 -> Best@8 57.9% -> Best@16 59.0%; each verifier alone adds "10+%" | 100 max steps, 64k context — [Source](https://huggingface.co/agentica-org/DeepSWE-Preview) |
| 2512.21919 SWE-RM | Execution-free generative RM (30B-A3B) scoring full trajectories, RM@32 | 32 | Trained RM on ~100k filtered trajectories (from 400k+) | At N=32: Qwen3-Coder-Flash 52.6% (Agentless EB) / 54.6% (DeepSWE-EB) / 62.0% (SWE-RM); OpenHands-LM-32B 42.4/44.2/48.8%; Qwen3-Coder-Max 65.0/67.6/74.6% | 32 rollouts + 32 RM passes — [Source](https://arxiv.org/html/2512.21919) |
| 2505.23604 Satori-SWE | Evolutionary refinement guided by RM, pooled candidates | 50 | RM + unit tests | 32B: 35.8% greedy -> 41.6% Best@50 | 50 candidates — [Source](https://arxiv.org/pdf/2505.23604) |
| 2503.23803 Thinking Longer, Not Larger | Internal (long CoT) + external (dev-scaled search) TTS on Qwen2.5-Coder-32B | — | — | 32B-class reaches 46% Verified (per paper title/abstract) | — [Source](https://arxiv.org/pdf/2503.23803) |
| 2505.18955 Co-PatcheR | 3 x 14B specialists (localize; generate+critique; repro tests with/without assertions) then majority vote | n/a | Execution on generated repro tests + majority vote | 46% Verified with 3x14B, vs prior 70B specialist 41% | lowest training cost among specialists — [Source](https://arxiv.org/abs/2505.18955) |
| 2507.23370 Trae Agent | Agent-based ensemble: generation, pruning, selection modules over three frontier LLMs | multiple | LM selector | 75.20% pass@1; +10.22 avg over ensemble baselines | frontier-model ensemble — [Source](https://arxiv.org/abs/2507.23370) |
| 2601.04171 Agentic Rubrics | Expert agent explores repo, writes context-grounded checklist; candidates scored against it | parallel TTS | Rubric LM judge, execution-free | 54.2% (Qwen3-Coder-30B-A3B), 40.6% (Qwen3-32B); >= +3.5 over strongest baseline | — [Source](https://arxiv.org/abs/2601.04171) |
| 2601.22129 SWE-Replay | Replace from-scratch resampling with branching from archived trajectories at rarely-explored, reasoning-dense steps; filter archive by repo regression tests only | 10 patches, Agentless-style selection | Regression tests + Agentless selection; no LM judge | mini-SWE-agent: Gemini-2.5-Pro 58.0 -> 60.2%, Devstral-Small-2 62.2 -> 63.2%; Live-SWE-agent Devstral 63.2 -> 65.0%; Multilingual 31.0 -> 38.0% | cost -8% to -17.4% vs naive 10-sample scaling; LLM-as-judge step selection was worse (54.0%) and 2.2x cost — [Source](https://arxiv.org/html/2601.22129) |
| 2604.16529 Scaling TTC for Agentic Coding (RTV+PDR) | Summarize each rollout; Recursive Tournament Voting (pairwise, V=8 votes) for selection; PDR refines new rollouts on K=4 selected summaries | N=16, T=2 (32 rollouts) | LM pairwise judge over structured summaries | RTV alone +5–6 pts on Verified (Sonnet 4.5 67.4 -> 73.6%); full recipe Opus 4.5 70.9 -> 77.6%, GPT-5 61.4 -> 69.8% | 32 rollouts + summaries + 8 votes x 15 comparisons — [Source](https://arxiv.org/html/2604.16529) |
| 2604.14820 SWE-TRACE HG-TTS | Step-level pruning: policy proposes K actions, rubric-conditioned PRM scores them, execute only top-B (B=1) | K per step | Trained rubric PRM | Qwen3-4B 38.9% greedy -> 40.2% parallel -> 40.7% HG-TTS; Qwen3-30B-A3B 63.5 -> 69.9 -> 71.2% | Latency/issue (min): 4B greedy 18.7 / parallel 54.1 / HG-TTS 31.6; 30B 22.4 / 63.8 / 36.5; env executions 67 / 376 / 119 (4B) — [Source](https://arxiv.org/html/2604.14820) |
| 2602.05242 EGSS | Entropy-guided stepwise scaling + test-suite augmentation | — | — | Kimi-K2 63.2 -> 72.2%, GLM-4.6 65.8 -> 74.6% | tokens -28% vs existing TTS — [Source](https://arxiv.org/abs/2602.05242) |
| 2608.22191 Disagree to Explore | Routing-guided TTS; select by agreement at informative patch positions across trajectories (patch self-consistency without string match) | — | None (consensus) | gpt-oss macro-avg 44.9% (uniform) -> 48.2% | no judge, no selection-time execution — [Source](https://arxiv.org/abs/2608.22191) |
| 2609.36371 LatentSift | Replace first-stage LM verifier with hidden-state filter before hybrid verification | 16 | Hidden-state probe + EB | DeepSWE 59.26 -> 60.06% Best@16 | EF-verifier tokens -66.6 to -81%; total verification tokens -49 to -62% — [Source](https://arxiv.org/abs/2609.36371) |
| 2609.31995 CER | Early terminal-reward prediction from prefixes with retrieved rubrics; beam over prefixes | RM@8 | LM judge with adaptive rubrics | +4.2 pp RM@8 (Nemotron 3 Ultra, 63.4 -> 67.6%), +2.0 pp (Qwen 3.6 27B, 69.4 -> 71.4%); matches best baseline with 15.3% / 74.2% of tokens | — [Source](https://arxiv.org/html/2609.31995) |
| 2610.34577 Rethinking Verification in Agent Harnesses | 17 harness designs: initial instructions, mid-turn reminders, completion gates, independent review | 1 (+review) | Separate LM critic reviewing the finished work | Independent review +3.9 to +6.6 pts on SWE-bench Verified at 1.00–1.04x baseline cost; +5.9 to +9.5 on Terminal-Bench at 1.13–1.25x; completion gates give larger gains but "may substantially increase cost" | ~1.0–1.04x — [Source](https://arxiv.org/abs/2610.34577) |
| 2602.03419 SWE-World | Verifier-ranked TTS@8 | 8 | Learned verifier | SWR-32B 55.0 -> 68.2% | 8x — [Source](https://arxiv.org/pdf/2602.03419) |
| 2602.03411 SWE-Master | TTS@8 | 8 | — | 32B 61.4 -> 70.8% (as tabulated in SWE-TRACE Table 2) | 8x — [Source](https://arxiv.org/html/2604.14820) |

### Inferences
- Best@8 is where most of the gain lives: R2E-Gym/DeepSWE Best@8 57.9% vs Best@16 59.0%; SWE-Master, SWE-World, Skywork all report at N=8. Going from 8 to 16+ buys 1–2 points for 2x cost.
- Execution-based selection with generated tests consistently beats LM-only judging at equal N when the policy is weak (R2E-Gym, Kimi-Dev), but trained RMs beat execution-based selection when the RM is strong (SWE-RM). The hybrid is the robust choice.
- Patch-level majority vote alone is weak (Agentless: 25.67% vs 32.00% with tests), but it is free.

### Gaps
- No paper reports resolve rate vs wall-clock minutes for a fixed GPU; all costs are tokens or dollars. SWE-TRACE Table 5 (latency in minutes/issue) is the only wall-clock table found, and it does not state hardware.
- SWE-Swiss: I found no arXiv paper or primary source for its TTS numbers.
- Trae Agent per-component numbers (number of candidate patches, selector details) are not in the abstract; the full paper was not read.

## (2) Verifier / critic / reranker evidence at small model scale

### Takeaway
A 14B LoRA execution-free verifier trained on ~5.7k trajectories (R2E-Gym) saturates at roughly the same Best@K as execution-based selection and is biased toward stylistic features; a 0.6B monitor trained only on 27B-policy trajectories can predict failure well enough to save 15–20% tokens (FailFast). No paper trains a verifier at 4B for patch selection; the closest evidence (R2E-Gym 14B, StopLoss Qwen3-8B, FailFast 0.6B) suggests a small critic is useful for pruning/stopping but not as the sole selector.

### Cited Findings
- R2E-Gym execution-free verifier: Qwen2.5-Coder-14B, LoRA rank 64, 2 epochs, lr 1e-5, trained on 5,700 balanced trajectories including on-policy samples from the 32B agent; score P(YES)/(P(YES)+P(NO)); Best@K plateaus ~42.8%; drops to 37.6% at Best@26 when shown only the final patch instead of the trajectory, which the authors read as reliance on agent thoughts; "biased and often relies on stylistic features" — [Source](https://arxiv.org/html/2504.07164)
- DeepSWE-Verifier is an SFT fine-tune of Qwen3-14B; the model card claims roughly +10% on SWE-bench Verified when paired with DeepSWE-Preview — [Source](https://friendli.ai/models/agentica-org/DeepSWE-Verifier)
- SWE-RM (30B-A3B) needed ~100k usable trajectories from 400k+ collected; the authors argue TTS performance alone is not sufficient to judge a verifier and that classification accuracy and calibration matter for RL — [Source](https://arxiv.org/html/2512.21919)
- Nebius critic was initialized from LLaMA-3.1-70B and trained on ~80k positive/negative trajectories — [Source](https://arxiv.org/html/2505.13652)
- SWE-Gym: verifiers trained on SWE-Gym trajectories lift fine-tuned agents to 32.0% Verified / 26.0% Lite — [Source](https://arxiv.org/abs/2412.21139)
- FailFast: a 0.6B monitor trained with terminal and dense fail-to-pass supervision on Qwen3.6-27B trajectories only; at 5% false-positive rate saves 14.6–20.4% execution tokens across policies including a closed API model; Restart-Smart (restart with the interrupted diff offered as an optional overlay) raises Qwen3.6-27B from 66.6% to 71.8% at 25% FPR, while cold restart gives only 66.8% — [Source](https://arxiv.org/abs/2608.03222)
- StopLoss: small fine-tuned monitor (Qwen3-8B backbone mentioned for SWE-bench Pro) trained on SWE-rebench trajectories labeled by verifier outcome; preserves 99% of successful runs while lowering cost per retained solve 4.5% on Verified and up to 31–48% on Pro; per-check cost 1–2 orders of magnitude below general-purpose judges — [Source](https://arxiv.org/abs/2610.25525)
- SWE-TRACE: rubric PRM trained from execution preference pairs (pass vs fail) plus rubric preference pairs judged by MiniMax 2.5; reused as step-level guide; on Qwen3-4B the HG-TTS gain is only +1.8 (38.9 -> 40.7%) versus +7.7 on 30B-A3B, and the authors note "the advantage is smaller on the 4B model" — [Source](https://arxiv.org/html/2604.14820)
- SWE-Replay found that LLM-as-a-Judge step selection was worse than its heuristic (54.0% vs 60.0%) and doubled cost ($3.31 vs $1.52) — [Source](https://arxiv.org/html/2601.22129)
- SWE-Shepherd trains a "lightweight" action-level PRM from SWE-Bench trajectories and reports difficulty aligning intermediate rewards with final task success — [Source](https://arxiv.org/abs/2604.10493)
- CER reports only ~60–65% pairwise agreement between rubric rewards and terminal rewards, "indicating that the two signals are not interchangeable" — [Source](https://arxiv.org/html/2609.31995)

### Inferences
- A Gemma 4 E4B LoRA critic trained on a few thousand of our own rollouts (pass/fail labels from the public train split) is plausible as a tie-breaker or early-stop monitor, following R2E-Gym's 5.7k-trajectory recipe, but expect ~43% ceiling-type behavior and stylistic bias; it should see the trajectory, not just the diff.
- Given FailFast's 0.6B result, a failure monitor is a cheaper and better-supported use of a small critic than best-of-N ranking.

### Gaps
- No published result for a verifier at or below 4B used for patch selection; R2E-Gym 14B is the smallest selection verifier found.
- No paper quantifies how much verifier accuracy degrades when trained on trajectories from a different policy/scaffold than the one evaluated (CER mentions sensitivity qualitatively).

## (3) Budget-aware agents: stopping, step budgets, when to submit

### Takeaway
Direct evidence that resolved runs finish early is strong (SWE-agent: 93% of resolved instances finished under budget vs 69% overall) and early-termination monitors cut 15–32% of cost with <=0.2 pt loss; the one Gemma 4 agent report found that without an explicit step-budget prompt the model "explores indefinitely" and never submits, that step_limit 30 was too tight and 100 worked, and that enabling thinking degraded patch quality. No controlled sweep of step caps on SWE-bench Verified was found.

### Cited Findings
- SWE-agent used a per-instance cost cap; 93% of resolved instances finished before hitting the budget vs 69% overall, and successful runs complete earlier and cheaper; the authors conclude a larger budget "probably would not substantially raise performance" — [Source](https://arxiv.org/html/2405.15793v2)
- SWE-smith notes 31 instances resolved after 40+ steps, so a tight cap cuts off some solvable tasks — [Source](https://arxiv.org/pdf/2504.21798)
- SWE-Shepherd evaluated on 100 Verified tasks with a 30-step limit; mini-SWE-agent reached 57% there with gpt-5-mini averaging ~15 steps — [Source](https://arxiv.org/pdf/2604.10493)
- SWE-TRACE: cascaded shortest-path SFT reduced average steps from 74 to 66 on Qwen3-4B while raising resolve rate 33.6 -> 37.8% and cutting total tokens 21.5%; greedy 4B runs averaged 67 environment executions and 18.7 min/issue — [Source](https://arxiv.org/html/2604.14820)
- Gemma 4 26B-A4B FP8 on SWE-bench Lite (38.67%, 116/300, mini-SWE-agent-style custom scaffold): default step_limit 250 led to over-engineering and never submitting; step_limit 30 cut the edit phase short; final step_limit 100 with a prompt "You have ~100 steps. Aim to:" splitting phases (1–15 explore/reproduce, 15–40 locate/edit, 40–60 verify/submit). Without the budget prompt the model "burns 100 steps without ever submitting". `--reasoning-parser gemma4` (thinking) "degraded patch quality" with whole-file rewrites and was disabled; temperature 0.3; the model "cannot reliably produce OpenAI-style JSON tool calls when there are 10+ tools registered" (OpenHands with 11 tools: 40+ errors/instance), so the author switched to a text backticks action format parsed by regex; 300 tasks took ~19 h with 2 workers — [Source](https://ai-muninn.com/en/blog/swe-bench-lite-gemma4-26b-38-percent)
- EET: experience-driven early termination cut total cost 19–55% (avg 32%), API calls -21%, with at most 0.2% resolve-rate loss across three SE agents; termination opportunities were found for 11% of issues — [Source](https://arxiv.org/abs/2601.05777)
- EarlyEval predicts final outcome from partial trajectories and eliminates 13–26% of agent steps across SWE-bench Verified, TerminalBench and Toolathlon — [Source](https://www.alphaxiv.org/abs/2609.02783)
- Nebius "Until Submitted" regime (retry up to 10 times if the agent fails to submit) raised the 72B policy from 16.2% to 22.8% with ~1.65 extra attempts per problem — [Source](https://arxiv.org/html/2505.13652)
- Rethinking Verification: completion gates (forcing self-check before finishing) gave larger gains than instructions/reminders but "may substantially increase cost"; independent review gave +3.9–6.6 pts on SWE at 1.00–1.04x cost — [Source](https://arxiv.org/abs/2610.34577)
- Gemma 4 thinking is a conversation-level on/off control token (<|think|> in the system instruction); larger models may emit a thought channel even when off; thoughts must not be stripped between function calls within a turn; recommended sampling temperature 1.0, top_p 0.95, top_k 64 — [Source](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4)
- DeepSWE evaluated with 100 max environment steps and 64k context; RL used a 5-minute test time limit and masked trajectories that timed out at 20 minutes — [Source](https://huggingface.co/agentica-org/DeepSWE-Preview)

### Inferences
- A phase-labelled step budget in the system prompt plus a hard max_tool_calls around 40–60 is the cheapest, best-supported budget mechanism for Gemma 4; the Lite report's 30-step failure suggests our ~6 min/task budget should be spent on fewer, denser steps rather than a tiny cap.
- A "forced submit" guard (if remaining time < X, emit the current diff) converts non-submissions into partial credit; Nebius shows non-submission is a large failure bucket for mid-size policies.
- Time-aware prompting ("you have N tool calls left") is cheap in ADK via a tool-call counter injected into the prompt; no paper tests it directly, but the Gemma 4 report and SWE-agent correlation support it.

### Gaps
- No controlled ablation of step cap (e.g., 15/30/50/100) vs resolve rate on SWE-bench Verified for any model; the Lite report is anecdotal (single configuration per setting).
- No evidence on thinking_budget values between 0 and full thinking for SWE agents; Gemma 4 officially exposes only on/off, and a Gemini-API client error ("Thinking budget is not supported for this model") is reported for Gemma 4 — [Source](https://bbs.pyvideotrans.com/show/3933)

## (4) Throughput and parallelism on 4x L4: concrete estimates

### Takeaway
One L4 measured with vLLM gives ~51 tok/s single-stream and ~193 tok/s at 4 concurrent for a 7B AWQ model; scaling by weight bytes implies ~25–45 tok/s for 31B int4 under TP4 (PCIe-bound), ~30–35 tok/s for 12B int4 on one GPU, and ~70–100 tok/s for E4B. At 6 min/task serial, a 31B agent without thinking gets roughly 15–25 steps; with 2k thinking tokens/step it gets ~5. Parallel rollouts of the same task are nearly free in decode throughput (batch 4 = 3.8x aggregate) and share prefill via prefix caching, so N=4 parallel rollouts cost ~1.3x wall-clock, not 4x, if KV memory allows.

### Cited Findings
- NVIDIA L4: 24 GB, 300 GB/s memory bandwidth, PCIe Gen4 x16 (64 GB/s), no NVLink listed, FP16 tensor ~121 TFLOPS dense (242 sparse), INT8/FP8 ~242 dense, 72 W — [Source](https://www.nvidia.com/en-us/data-center/l4/)
- Measured, one L4, vLLM 0.29.0, Qwen2.5-7B, 512 in / 256 out: output tok/s at concurrency 1/4/16/32/64 = BF16 17.5/67.2/222.4/386.9/589.3; FP8 28.9/112.0/382.5/668.8/955.8; AWQ 4-bit 50.8/193.0/534.9/862.8/1080.9; AWQ p95 TTFT 175 ms at c=1, 681 ms at c=4; AWQ GSM8K -0.4 pts vs BF16 — [Source](https://github.com/Aman12x/vllm-serving-bench)
- RunPod lists L4 at ~17 t/s per-user decode and ~102 t/s aggregate (model unspecified) — [Source](https://www.runpod.io/articles/guides/best-gpu-for-llm-inference)
- Qwen official vLLM speed benchmark: Qwen2.5-32B-Instruct AWQ ~51.9 tok/s at short input, ~18.67 GB memory (GPU not L4; batch 1) — [Source](https://qwen.readthedocs.io/en/v2.5/benchmark/speed_benchmark.html)
- vLLM automatic prefix caching hashes full KV blocks (parent hash + tokens + extras such as LoRA ID) and reuses them for any request sharing the prefix; "almost a free lunch and won't change model outputs"; only full blocks are cached — [Source](https://docs.vllm.ai/en/latest/design/prefix_caching.html)
- Decode at low batch (1–4) is dominated by reading weights once per step, so aggregate throughput scales ~linearly with batch until compute limits — [Source](https://www.spheron.network/blog/l40s-vs-a100/)
- DGX Spark Gemma 4 26B-A4B run used vLLM with --max-model-len 96000, ~27 GB weights, ~100 GB KV; no tok/s reported — [Source](https://ai-muninn.com/en/blog/swe-bench-lite-gemma4-26b-38-percent)

### Inferences (back-of-envelope; all estimates, not measurements)
- Effective decode bandwidth implied by the measured 7B AWQ figure is ~90 GB/s of weight traffic per second (4.4 GB x 50.8/s), i.e., ~30% of L4 peak, consistent with dequant-kernel overhead at batch 1. Scaling by weight bytes: 12B int4 (~7 GB) on one L4 -> ~30–35 tok/s; E4B int4 (~2.5–3.5 GB) -> ~70–100 tok/s; 31B int4 (~16–17 GB) does not fit on one L4 with KV, so TP2 or TP4 is required; TP4 reads ~4.2 GB/GPU/step (~14 ms) plus two PCIe all-reduces per layer, giving ~25–45 tok/s single-stream, with TP overhead likely dominating for 12B/E4B.
- Consequence for serial 6 min/task: at ~30 tok/s and ~400 output tokens/step (no thinking), one step is ~13 s plus tool latency, so ~15–25 steps fit; at 2,000 thinking tokens/step it is ~80 s/step, ~4–5 steps. Thinking must be off or tightly budgeted for 31B; 12B/E4B can afford ~2x the steps.
- Parallel rollouts: measured c=4 gives 3.8x aggregate output, so N=4 rollouts of the same task finish in ~1.05–1.3x the wall-clock of one rollout if KV fits; KV for 32k-token contexts on a 31B dense model is several GB per sequence (exact Gemma 4 KV layout not found), so ~6–8 concurrent 32k sequences after ~17 GB weights on 96 GB is the practical ceiling for 31B; 12B/E4B allow 16+.
- Prefix caching matters most for the agent loop itself (each step re-sends the full history); it also shares the system prompt + issue + initial repo context across parallel rollouts of one task, but diverges after the first tool call.
- For 12B/E4B, data-parallel replicas (2–4 vLLM instances, TP1 or TP2) will beat TP4 for aggregate rollout throughput because PCIe all-reduce latency is a fixed per-step tax. This is a tuning recommendation, not a measured claim.

### Gaps
- No measured vLLM throughput for any Gemma 4 size on L4 or for 4x L4 TP4 was found despite three searches; the only L4 measurement is Qwen2.5-7B.
- No source states Gemma 4 KV-cache bytes per token for 31B/12B/E4B.
- Whether vLLM n>1 sampling shares the prompt KV across samples in V1 is not stated in the design doc fetched (it describes identical-prefix reuse across requests, which covers the case).

## (5) Most promising ideas for us, ranked by expected gain per minute of eval wall-clock

### Takeaway
Under a serial ~6 min/task budget the ordering is: free selection signals first (regression tests, submit guard, budget prompt), then cheap independent review, then N=2–4 parallel rollouts only if the scorer runs tasks serially and KV allows, with execution-based selection on the repo's own tests plus a scratch repro test.

### Cited Findings
1. Phase-labelled step budget in the system prompt + hard max_tool_calls (~40–60) + forced-submit guard. Evidence: Gemma 4 explores indefinitely without a budget prompt and never submits; step_limit 100 with phase prompt worked, 30 did not — [Source](https://ai-muninn.com/en/blog/swe-bench-lite-gemma4-26b-38-percent); 93% of SWE-agent resolved runs finished under budget — [Source](https://arxiv.org/html/2405.15793v2); non-submission retry added +6.6 pts for a 72B policy — [Source](https://arxiv.org/html/2505.13652). Cost: ~0.
2. Thinking off (or tiny budget) for 31B; spend tokens on tool calls. Evidence: thinking mode degraded Gemma 4 patch quality in the only agent report — [Source](https://ai-muninn.com/en/blog/swe-bench-lite-gemma4-26b-38-percent); TTI shows interaction scaling is a complementary axis to per-step reasoning for a Gemma 3 12B agent — [Source](https://arxiv.org/abs/2506.07976). Cost: negative (saves ~60–80% of decode time per step at 31B speeds).
3. Regression-test gate before submit using only the repo's existing tests (Agentless-style LLM-filtered set), run via run_command on the files touched by the diff. Evidence: regression-only selection reaches 47.4% vs 51.0% for the full hybrid in R2E-Gym — [Source](https://arxiv.org/html/2504.07164); SWE-Replay filters its archive by regression tests alone and finds it necessary — [Source](https://arxiv.org/html/2601.22129); Agentless +1.3 pts from regression tests at $0.01 — [Source](https://arxiv.org/html/2407.01489). Cost: one test run (<= 300 s cap; target the touched modules).
4. Independent review sub-agent (ADK agent_tool) that reads the final diff + issue and may send the agent back once. Evidence: +3.9–6.6 pts on SWE-bench Verified at 1.00–1.04x cost across seven frontier models — [Source](https://arxiv.org/abs/2610.34577). Cost: one extra LLM call; caveat: measured on frontier models, not 4–31B.
5. Scratch repro test in /tmp written before the fix, then executed after; keep the patch only if fail-to-pass and regressions pass. Evidence: Kimi-Dev 3x3 self-play beats 40-way majority vote; reproduction tests give the largest selection gain in Agentless (+5 pts) — [Source](https://arxiv.org/html/2509.23045), [Source](https://arxiv.org/html/2407.01489). Failure mode: only 94/300 Agentless repro tests actually flipped on the gold patch, so treat it as a filter, not an oracle. Cost: ~2–4 tool calls.
6. N=2–4 parallel rollouts per task (ParallelAgent) with hybrid selection: regression score, then repro-test score, then patch-position agreement as tie-break. Evidence: Best@8 captures most of Best@16 (57.9 vs 59.0%) — [Source](https://huggingface.co/agentica-org/DeepSWE-Preview); consensus at informative patch positions gives +3.3 with no judge — [Source](https://arxiv.org/abs/2608.22191); measured c=4 is 3.8x aggregate throughput on L4 — [Source](https://github.com/Aman12x/vllm-serving-bench). Cost: ~1.1–1.3x wall-clock if the scorer is serial and KV fits; ~Nx if the scorer already runs tasks concurrently.
7. Replay-from-prefix instead of from-scratch retry when the first rollout fails or hits the budget without submitting (branch at the last reasoning-dense step before the first edit, carrying the diff). Evidence: SWE-Replay -8 to -17% cost with equal or better resolve rate; Restart-Smart with diff overlay 66.6 -> 71.8% vs cold restart 66.8% — [Source](https://arxiv.org/html/2601.22129), [Source](https://arxiv.org/abs/2608.03222). Cost: a second partial rollout only on failures.
8. E4B LoRA failure monitor (train on our own train-split rollouts with pass/fail labels) to abort hopeless rollouts at ~50% of budget and trigger idea 7. Evidence: 0.6B monitor saves 14.6–20.4% tokens at 5% FPR — [Source](https://arxiv.org/abs/2608.03222). Cost: a few hundred rollouts of labelled data; lowest priority given deadline.

### Inferences
- Ideas 1–3 are harness/YAML changes with no training and should be in the baseline before any model comparison; idea 6 is the only one whose cost depends on unknown scorer concurrency and should be gated on a measured vLLM concurrency test.

### Gaps
- No source measures any of these at Gemma 4 scale on SWE-bench; all per-minute estimates derive from section 4 inferences.

## (6) Not feasible and why

### Takeaway
Anything requiring 10+ full rollouts per task, a 14B+ trained verifier, MCTS over tool actions, or test-time self-play at 40x40 is out of budget at 6 min/task on 4x L4; summary-based tournament voting is possible only at N<=4.

### Cited Findings
- CodeMonkeys-style 10 state machines x 8 iterations cost ~$2.3k with Claude and 69.8% coverage needed a strong selector to realize 57.4% — [Source](https://arxiv.org/html/2501.14723); Kimi-Dev needs 40 patches, 40 tests and 1,600 test executions per issue — [Source](https://arxiv.org/html/2509.23045); SWE-RM uses RM@32 and a 30B-A3B RM trained on ~100k trajectories — [Source](https://arxiv.org/html/2512.21919); Nebius critic is 70B trained on ~80k trajectories — [Source](https://arxiv.org/html/2505.13652); RTV+PDR uses 32 rollouts plus 8 votes per comparison — [Source](https://arxiv.org/html/2604.16529); SWE-TRACE parallel rollout took 54–64 min/issue and HG-TTS 32–37 min/issue even on 4B/30B-A3B — [Source](https://arxiv.org/html/2604.14820); SWE-Search MCTS relies on Moatless-style templated tools and LM value estimates, which SWE-Replay reports do not generalize to free-form bash agents and are miscalibrated — [Source](https://arxiv.org/html/2601.22129).

### Inferences
- Full-trajectory best-of-N beyond N=4, trained 14B+ verifiers, MCTS/lookahead with K candidate actions per step, and 40x40 self-play each exceed the 6 min/task budget by 5–10x at 31B decode speeds; SWE-TRACE's own step-level pruning still took 31.6 min/issue on a 4B model.
- A verifier larger than the policy (R2E-Gym 14B for a 32B policy is already "small") is not affordable in 96 GB alongside a 31B int4 policy at 32k context; only an E4B LoRA critic sharing the base weights is plausible.
- Agentic Rubrics (expert agent explores the repo to write a checklist) duplicates the exploration cost of a rollout and is unproven below 30B-A3B.

### Gaps
- Scorer concurrency is unknown; if the scorer already runs several tasks in parallel, even N=2 rollouts may not fit, and all parallel ideas collapse to the serial list (ideas 1–5, 7).
