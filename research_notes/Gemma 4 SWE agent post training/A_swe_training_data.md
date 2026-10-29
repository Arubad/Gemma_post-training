# Cluster A: SWE-agent training data, synthetic task generation, execution environments (2024–Oct 2026)

Scope: what we could reuse or re-implement to post-train Gemma 4 (E4B / 12B / 31B QAT-int4, LoRA r<=128, Kaggle L4x4, no internet in GPU sessions, <$50 teacher API) into a repo-level Python bug-fixing agent evaluated in a 9-tool ADK harness, 6 min/task, ~120 hidden tasks from private repos. Researched 2026-10-29. Every number below is from the cited source; items I could not verify are in the Gaps lists.

Compute framing used throughout (my assumption, not sourced): 2–3 accounts x 15 h/week x ~5 weeks to Dec 2 = roughly 150–225 L4x4-node-hours total, including evaluation.

---

## (1) Paper / dataset table

### Takeaway
Every open result that moves a <=8B model from single digits to 30–42% SWE-bench Verified does it with 5k–18k *successful* teacher trajectories and full fine-tuning at 32k–128k context. Tens of thousands of permissively licensed trajectories (OpenHands and SWE-agent schemas, native function-calling) are already on Hugging Face, so for us the binding constraint is tool-schema translation, sequence length and GPU-hours, not trajectory supply.

### Cited Findings

| System | arXiv | Method (1–2 lines) | Data scale | Model trained | SWE-bench Verified / Lite | License | Trajectories on HF? Schema |
|---|---|---|---|---|---|---|---|
| SWE-Gym | 2412.21139 | First real-task training env: real GitHub issues with executable envs; SFT + trained verifier | 2,438 real Python tasks, 11 repos; 491 successful trajectories (gpt-4o + Claude-3.5-Sonnet) | Qwen2.5-Coder 7B/14B/32B full FT, 32,768 ctx | 7B 1.8→10.6, 14B 4.0→16.4, 32B 7.0→20.6 Verified; Lite 7B 1.0→10.0, 32B 3.0→15.3; 32B + verifier Best@16 = 32.0 V / 26.0 L | not verified | Yes: SWE-Gym/OpenHands-Sampled-Trajectories, 6,055 total / 491 successful, OpenHands, function calling ([Nebius table](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b)) |
| SWE-smith | 2504.21798 | Given any Python repo, build env, inject bugs (procedural AST, LM-modify, LM-rewrite, PR-mirror, combine), keep those that break tests; LM writes issue text | 50,137 candidates → 50.1% break a test; 128 repos; 5,016 Claude-3.7-Sonnet trajectories (36% resolve, 17,906 attempts) | Qwen2.5-Coder-32B (5k traj), 7B (2k traj) | 32B 40.2 V / 30.7 L; 7B 15.2 V | MIT (trajectories) | Yes: SWE-bench/SWE-smith-trajectories, splits `tool`/`xml`/`ticks` (3 action formats), fields messages/instance_id/resolved/patch; Nebius lists 49,897 total / 21,513 successful ([HF card](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)) |
| R2E-Gym | 2504.07164 | SYNGEN: back-translate commits into issues + generate tests; hybrid execution/execution-free verifier for TTS | 8.7k tasks; 3,231 Sonnet-3.5 SFT trajectories | 32B | 34.4 V pass@1; 51 V with hybrid verifier (each verifier alone saturates 42–43) | not stated on card | Yes: R2E-Gym/R2EGym-SFT-Trajectories, 3,231 rows, `messages` list, function-definition prompt ([HF card](https://huggingface.co/datasets/R2E-Gym/R2EGym-SFT-Trajectories)) |
| SWE-Dev (THUDM) | 2506.07636 | Scale task instances from PRs, synthesize fail-to-pass tests, OpenHands trajectories from DeepSeek-V3, RFT/KTO/OREO | 38k instances / 4,413 repos; 17k unevaluated + 2.3k correct trajectories | Qwen2.5-Coder 7B/32B, Llama-3.1-8B, GLM-4-9B | 7B 23.4, 8B 18.0, 9B 13.6, 32B 36.6 V | not stated | HF THUDM/SWE-Dev-32B links data; schema not verified |
| SWE-Dev (feature-driven) | 2505.16975 | Feature-development benchmark/training set (not bug fixing) | – | – | no Verified numbers | – | – |
| SWE-Synth | 2504.14757 | LLM re-implements a masked function/class (Qwen2.5-Coder-32B, T=0.7); keep variant if ≥1 test fails; agent rollouts give fix trajectories | 21,804 variants → 9,459 verified (43.4%); 7 SWE-Gym repos; 3,018 patched versions | Qwen2.5-Coder 7/14/32B, LoRA r=8 (32B QLoRA) | 32B, 2,136 traj: 22.2 V vs 18.2 for same-size SWE-Gym real data; Lite 24.33 vs 23.67 | not stated | GitHub only (FSoft-AI4Code/SWE-Synth) |
| SWE-Flow | 2506.09003 | TDD: derive incremental dev steps from unit tests via runtime dependency graph | 16,061 train / 2,020 test | open models (sizes in paper) | no Verified numbers (own SWE-Flow-Eval) | not checked | not checked |
| SWE-rebench V1 | 2505.20411 | Automated pipeline: real PRs → executable tasks; continuous decontaminated benchmark | 21,000+ Python tasks | (dataset) | – | – | Task dataset nebius/SWE-rebench |
| Nebius OpenHands trajectories | (blog) | Qwen3-Coder-480B on OpenHands v0.54 over SWE-rebench | 67,074 traj / 32,161 successful / 3,792 issues / 1,823 repos; avg 64 turns | Qwen3-30B-A3B-Instruct, Qwen3-235B | 30B-A3B 25.7→50.3 V (pass@5 68.3); 235B 61.7 V | CC-BY-4.0 data, Apache-2.0 models | Yes: nebius/SWE-rebench-openhands-trajectories; roles system/user/assistant/tool, native tool_calls, args as serialized strings, no truncation ([blog](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b)) |
| Nebius SWE-agent trajectories | (blog, Dec 2024) | Various open action generators on SWE-agent | 80,036 traj / 13,389 resolved / 1,202 repos | 72B-class agent + critic | card says 40.6% | CC-BY-4.0 (+Llama 3.1 terms) | Yes: nebius/SWE-agent-trajectories; fields trajectory/exit_status/generated_patch/eval_logs; text actions, no function calling ([HF card](https://huggingface.co/datasets/nebius/SWE-agent-trajectories)) |
| SWE-rebench V2 | 2602.23866 | Language-agnostic setup agent + LLM-judge ensemble | 32,079 tasks / 20 langs / 3,617 repos (+120k lightly validated) | – | – | not checked | Tasks only |
| Skywork-SWE | 2506.19290 | Incremental pipeline 146,568 PR instances → 23,389 installable → 10,169 validated; multi-teacher trajectories | 10,169 tasks / 2,531 repos; 8,209 trajectories | Qwen2.5-Coder-32B full FT, 32k ctx, 3 epochs, 8xH800 ~12 h | 38.0 V pass@1; 47.0 with TTS N=8; log-linear in #traj | model released; data not confirmed | Not confirmed |
| SWE-Swiss | no arXiv ID found; ACL Findings 2026 | Multi-task SFT (localization, repair, test generation) + RL on repair with test feedback | – | 32B | 60.2 V (author-page snippet) | – | not found ([ACL](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.790/)) |
| Kimi-Dev | 2509.23045 | Agentless-style skill-prior training, then 5k public trajectories SFT for agent mode | 5k public trajectories | 72B | 60.4 V (Agentless); 48.6 V (SWE-agent) | – | uses public trajectories |
| Nemotron-CORTEXA | no arXiv ID found; ICML 2025 PMLR v267 | Inference scaffold: better localization + solution diversity | – | – | – | – | n/a ([PMLR](https://proceedings.mlr.press/v267/sohrabizadeh25a.html)) |
| SWE-Fixer | 2501.05040 | BM25 + retriever model → editor model; 2 calls/instance | 110k issue–patch pairs | 72B | 30.2 V / 22.0 L; 32.8 / 24.7 with P2P filter | not checked | Dataset, not trajectories |
| Agentless | 2407.01489 | Localize → repair → validate, no agent loop | – | (GPT-4o) | 32.00 L, $0.70/task | MIT (known) | n/a |
| NL2Fix | 2304.03816 | NL bug description → Java fix on Defects4J | 283 Java bugs | – | n/a | – | n/a (not relevant) |
| SWE-Factory | 2506.10954 | SWE-Builder multi-agent env construction; exit-code F2P parsing (F1 0.99) | 337 valid of 671 issues (≈50%), $0.047/instance with GPT-4.1-mini | – | – | – | Pipeline only |
| BugPilot (FeatAdd) | 2510.19898 | Agent adds a feature; breakages of existing tests become bugs | 785 tasks / 86 repos | Qwen3-32B, 14B full FT, 32k ctx | 32B: +1.2k traj → 51.93 vs R2E-Gym 50.73, SWE-smith 50.53, base-mix 49.87; 14B 41.1 SFT → 42.2 RL, 45.3 all-data | not stated | not stated |
| SWE-Mirror | 2509.08724 | Mirror a real issue into another repo with a Gym env | 60,671 tasks / 40 repos / 4 langs; 12k trajectories | Qwen2.5-Coder 7B/32B, OpenHands | 7B 22.8, 32B 52.2 V (per Kwai-Klear table) | not checked | not checked |
| SWE-Lego | 2601.01426 | Hybrid real (SWE-rebench) + synthetic (SWE-smith-style) tasks; Qwen3-Coder-480B/OpenHands teacher; error masking + length curriculum | 32,119 tasks (18,409 real / 13,710 synth); 14,110 valid traj (+semi-resolved → 18.1k) | Qwen3-8B, 32B full FT, 128k ctx (YaRN), 4 epochs | 8B 7.6→42.2 V (49.6 TTS@16); 32B 23.2→52.6 (58.8 TTS) | not stated; "will be open-sourced" | Not confirmed |
| Klear-AgentForge | 2511.05951 | 66k mini-swe-agent-plus (bash-only) trajectories on SWE-smith tasks; SFT+RL+merge | 65,994 traj | Qwen3-8B | SFT-only 39.0 V; log-linear 1k→66k | MIT | Yes: Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k, THOUGHT + one bash block ([HF card](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k)) |
| SERA (Ai2) | 2601.20789 | Soft Verified Generation: teacher edits a random function, writes a PR text, re-solves it; line-recall between the two patches replaces tests | 200k+ traj from 121 repos; GLM-4.5-Air / GLM-4.6 teacher; vanilla SWE-agent (view/edit/submit/bash) | Qwen3-8B, 32B full FT, 3 epochs, 32k | 8B 31.7 V; 32B 49.5 (54.2 at 64k); 16k traj 47.07, 25k 49.53 | Apache-2.0 | Yes: allenai/Sera-4.6-Lite-T2 (25k) ([model card](https://huggingface.co/allenai/SERA-8B)) |
| SWE-Protégé | 2602.22124 | 7B learns when to ask an expert (~4 calls/task, 11% of tokens); SFT 4.8k + GRPO | 4.8k Sonnet-3.7 traj on SWE-smith tasks | Qwen2.5-Coder-7B | 42.4 V with Opus 4.1 expert; SFT-only ≈30.6 | – | – |
| SWE-Master | 2602.03411 | Multi-teacher (MiniMax-M2, GLM-4.6) SFT on R2E-Gym scaffold, 80k ctx, then RL; summary context manager | ~12.6k final traj from SWE-Gym/SWE-rebench/R2E-Gym | Qwen3-4B-Instruct, Qwen2.5-Coder-32B | 4B SFT 27.6 → RL 33.4; 32B 6.2 → 57.8 SFT → 61.4 RL (70.8 TTS@8) | not stated | GitHub only |
| SWE-Zero/SWE-Hero (NVIDIA) | 2604.01496 | 300k execution-free + 13k execution-verified trajectories from Qwen3-Coder-480B | 34,269 traj / 11,766 issues released | Qwen2.5-Coder 32B | 62.2 V; 44.1 Multilingual zero-shot | CC-BY-4.0 | Yes: nvidia/SWE-Hero-openhands-trajectories; OpenHands, roles system/user/assistant/tool ([HF card](https://huggingface.co/datasets/nvidia/SWE-Hero-openhands-trajectories)) |
| Nemotron-SWE-v1 / SFT-SWE-v2 / Cascade-SFT-SWE | – | NVIDIA OpenHands trajectories + Agentless-style localization/repair/test JSONL | v1: 59k traj (Qwen3-Coder-480B); v2: ~46k traj + ~210k agentless | – | – | CC-BY-4.0 (+BSD-3 subsets) | Yes ([HF](https://huggingface.co/datasets/nvidia/Nemotron-SWE-v1)) |
| OpenSWE / daVinci-Env | 2603.13023 | 45,320 Docker envs / 12.8k repos; $1.47M total; difficulty filtering | ~13k curated traj from ~9k envs | 32B / 72B | 62.4 / 66.0 V | README AGPL-3.0; HF tag mixed; gated | Gated ([HF](https://huggingface.co/datasets/GAIR/OpenSWE)) |
| Scale-SWE | 2602.09892 | 6M PRs → 100k verified instances / 5,200 repos | 71,498 traj | Qwen3-30B-A3B | 64 V | – | "will be public" |
| Open-SWE-Traces | 2606.16038 | 207,489 traj, 9 languages, MiniMax-M2.5 (thinking) + Qwen3.5-122B (non-thinking) over SWE-rebench-V2 | 207k traj / 20k PRs | Qwen3-30B-A3B | 61.7 V | permissive source repos | stated released |
| SWE-Fuse | 2603.07927 | Mix issue-guided and issue-free trajectories; entropy-aware RLVR | – | 8B, 32B | 8B 43.0, 32B 60.2; TTS@8 49.8 / 65.2 | – | – |
| SWE-Ladder | 2610.33667 | Stage-wise privileged self-distillation (localize/reproduce/patch) + GRPO | – | Qwen3-4B-Instruct-2507 | 9.6→16.4 (R2E-Gym scaffold); 13.0→17.6 (OpenHands) | – | – |
| SWE-at-Scale | 2610.21359 | Invert historical fixes into bug-induction patterns; up to 200 distinct bugs per file | – | none trained | – | – | – |
| Socratic-SWE | 2606.07412 | Self-evolving: distill own traces into skills → targeted synthetic repair tasks | – | – | 50.40 V after 3 iterations | – | – |

### Inferences
- The OpenHands and SWE-agent schemas dominate; nothing on HF is in our 9-tool schema, so any reuse requires a deterministic re-mapping (execute_bash→run_command, str_replace_editor view/create/str_replace→read_file/write_file/edit_file, finish→submit_patch). The code-graph tools (get_code_neighbors, search_similar_code, get_code_subgraph) have zero trajectory coverage anywhere.
- Across papers, SWE-agent-LM-7B sits at 15.2–17.0 and SWE-smith-trained 32B at 32.6–40.2 depending on scaffold ([SERA table](https://huggingface.co/allenai/SERA-8B); [SWE-Protégé](https://arxiv.org/abs/2602.22124)), so scaffold mismatch alone can cost ~8 points; this is a direct warning for training on OpenHands-shaped data and evaluating in ADK.

### Gaps
- Could not find arXiv IDs for SWE-Swiss (ACL Findings 2026 only) or Nemotron-CORTEXA (PMLR only).
- Licenses for SWE-Gym, R2E-Gym, SWE-Dev, SWE-Synth, Skywork, SWE-Lego, SWE-Mirror data were not stated on the pages read.
- Nebius SWE-agent-trajectories card says 40.6% "Verified"; I recall the Dec-2024 blog reporting Lite — unresolved.

---

## (2) Teacher-free synthetic task generation: what works

### Takeaway
Procedural/teacher-free bug injection is nearly free (0.00¢ per candidate) and yields 40–50% test-breaking tasks; downstream it trains about as well as LM-rewrite or PR-mirror bugs, and synthetic tasks give 2–3x higher trajectory yield than real issues because they are easier. Real-vs-synthetic head-to-heads at matched size show synthetic equal or slightly better.

### Cited Findings
- SWE-smith yields (candidates → break ≥1 test): Combine 10,092 → 96.9%; LM-Modify 17,887 → 56.0%; Procedural AST 15,641 → 40.2% (0.00¢); LM-Rewrite 4,173 → 35.0% (3.93¢); PR-Mirror 2,344 → 33.8% (5.53¢); overall 50.1%, 2.32¢ avg — [SWE-smith 2504.21798](https://arxiv.org/html/2504.21798)
- Downstream at 1,000 instances per source (≈500–800 traj): PR-Mirror 9.2±1.7, LM-Rewrite 8.8±1.7, Procedural 8.6±1.8, LM-Modify 5.7±1.5 Verified; LM-written issue text 7.7 vs original issue text 7.8, fixed template 6.4 — [SWE-smith](https://arxiv.org/html/2504.21798)
- SWE-Synth: 21,804 LLM re-implementations → 9,459 (43.4%) break a test; re-implementer size barely matters (3B 57.28% usable vs 14B 57.66%; 0.5B 9.00%); 32B trained on 2,136 synthetic traj: 22.2 V vs 18.2 with same-size SWE-Gym real traj; at 1k traj / 14B the gap is within noise (11.0 vs 11.4) — [2504.14757](https://arxiv.org/html/2504.14757)
- SWE-Lego: valid-trajectory yield ≈27% for real-world instances vs ≈66% for synthetic (5,010 of 18,409 vs 9,100 of 13,710); hybrid dataset gives +25.6 points over base on Qwen3-32B; adding "semi-resolved" trajectories +1.2 — [2601.01426](https://arxiv.org/html/2601.01426)
- BugPilot FeatAdd (agent adds a feature, breaking tests): +1.2k traj → 51.93 vs SWE-smith 50.53, R2E-Gym 50.73 at Qwen3-32B; abstract claims parity with half the data (1.2k vs 3k) — [2510.19898](https://arxiv.org/html/2510.19898)
- SERA: no unit tests at all; line-recall thresholds r=0/0.25/0.75/1.0 perform similarly (Kruskal-Wallis p=.066); hard-verified SERA data 30.00±1.41 vs SWE-smith 25.27±0.61 at matched Claude-3.7 teacher — [2601.20789](https://arxiv.org/html/2601.20789)
- SWE-Zero: 300k execution-free trajectories as stage 1, 13k execution-verified as stage 2 → 62.2 V at 32B — [2604.01496](https://arxiv.org/abs/2604.01496)
- SWE-Dev: filtering kept ~65% of trajectories with nearly identical performance to unfiltered — [2506.07636](https://arxiv.org/html/2506.07636)
- R2E-Gym SYNGEN: back-translate commits to issues + generate tests, 8.7k tasks, 34.4 V at 32B — [2504.07164](https://arxiv.org/abs/2504.07164)
- SWE-Factory: ~50% of issues become valid tasks at $0.047 each with GPT-4.1-mini; exit-code F2P parsing F1 0.99 — [2506.10954](https://arxiv.org/abs/2506.10954)
- SWE-at-Scale: inverting historical fixes yields diverse bugs, scaling to 200 generations per file (no training numbers) — [2610.21359](https://arxiv.org/abs/2610.21359)

### Inferences
- For us, procedural AST mutation plus "revert a real fix commit" (PR-mirror without the LM step) on the 129 public tasks' repos and any pip-installable Python repos with test suites is the only fully offline, zero-API task source, and the SWE-smith numbers say it is not meaningfully worse than LM-generated bugs.
- Test-pass filtering of *trajectories* appears to matter less than assumed (SERA, SWE-Dev, SWE-Lego semi-resolved), which legitimises using our own un-verified Gemma-31B rollouts as SFT data when test execution is slow.

### Gaps
- No paper isolates "revert-PR" alone from PR-mirror (which uses an LM to port the diff).
- No controlled study of synthetic-task training on *out-of-distribution private repos*, which is our hidden test condition.

---

## (3) Data-scale findings for small (4B–32B) models

### Takeaway
Gains appear with ~500 trajectories (7B: +9 points; 32B: +13) and grow roughly log-linearly: 7B goes 13.0% at 574 → 22.8% at 16.6k; 8B reaches 39–42% at 14–66k. Below ~2k trajectories a 7B model stays under ~16%, and LoRA-only results are consistently weaker than full fine-tuning.

### Cited Findings
- SWE-Gym, 491 traj, full FT 32k ctx: 7B 1.8→10.6, 14B 4.0→16.4, 32B 7.0→20.6 Verified; gains "particularly for the 32B model" — [2412.21139](https://arxiv.org/html/2412.21139)
- SWE-Dev scaling: 7B 13.0% at 574 traj → 22.8% at 16,639 traj, near-linear in log-log; RFT 1,000 traj 16.2 → 2,000 traj 18.0 → 2,300 traj 21.2 — [2506.07636](https://arxiv.org/html/2506.07636)
- SWE-smith: 7B on 2k traj 15.2; 32B on 5k traj 40.2; repo-count effect roughly logarithmic at 700 traj — [2504.21798](https://arxiv.org/html/2504.21798)
- Skywork-SWE: 32B log-linear up to 8k traj, no saturation; passes SWE-Dev-32B at N=2,000, OpenHands-LM at N=6,000, SWE-agent-LM at N=8,000 — [2506.19290](https://arxiv.org/html/2506.19290)
- SWE-Lego: Qwen3-8B 7.6 → 40.4 (14.6k) → 41.0 (14.1k filtered) → 42.2 (18.1k incl. semi-resolved); error masking + length curriculum worth +2.8 (8B) / +3.8 (32B); LR 1e-4 for 8B, 4 epochs, 128k seq — [2601.01426](https://arxiv.org/html/2601.01426)
- Klear: Qwen3-8B SFT 39.0 with 66k bash-only traj, solve rate ≈ linear in log(data) from 1k to 66k — [HF card](https://huggingface.co/datasets/Kwai-Klear/SWE-smith-mini_swe_agent_plus-trajectories-66k)
- SERA-8B 31.7 V from 25k traj; 32B 16k→47.07, 25k→49.53; removing reasoning traces drops 41.0→23.0; per-repo specialisation needs ~8k traj (α=1.0 matches teacher at 8k, α=0 does not even at 16k); 40 GPU-days for data+training of 32B — [2601.20789](https://arxiv.org/html/2601.20789)
- SWE-Master-4B (Qwen3-4B-Instruct-2507): SFT 27.6 → RL 33.4 with ~12.6k traj at 80k ctx — [2602.03411](https://arxiv.org/html/2602.03411)
- SWE-Ladder Qwen3-4B: 9.6→16.4 (R2E-Gym), 13.0→17.6 (OpenHands) via staged self-distillation + GRPO — [2610.33667](https://arxiv.org/abs/2610.33667)
- SWE-MeM: 4B model 43.4 V with memory-tool training — [2606.28434](https://arxiv.org/abs/2606.28434)
- LoRA evidence: SWE-Synth LoRA r=8 gives only +3.3 on 7B Lite (4.67→8.00) and ~0 on 32B (17.33→17.00) at 1k traj — [2504.14757](https://arxiv.org/html/2504.14757); SWE-Eff-14B LoRA on Qwen3-14B with ~3k R2E-Gym traj → 21.6 V (R2E scaffold, 32k, no function calling) — [Friendli listing](https://friendli.ai/models/ubicloud/SWE-Eff-14B); a LoRA study on Qwen2.5-Coder-7B found resolve rates near zero and used CE-loss proxies; doubling 500→1,000 traj cut CE by 12.7%, quality filtering only mattered at 2k (+3.6%, p=.016) — [2607.17205](https://arxiv.org/abs/2607.17205); SWE-Gym's MoatlessTools 32B used LoRA r=64, LR 5e-4, 5 epochs — [2412.21139](https://arxiv.org/html/2412.21139); TRACE trains LoRA adapters via RL for +15.0 V — [2604.05336](https://arxiv.org/abs/2604.05336)
- Action looping in SFT-only 7B: 31.0% of trajectories contain a repeated-action run >10 steps (SWE-agent-LM-7B 33.6%, 32B 24.4%); RL cut it to 0.8% and mean steps from ~60 to ~20 — [2602.22124](https://arxiv.org/html/2602.22124)
- Interaction budget: SWE-Dev-32B 34.0 at 30 rounds vs 36.6 at 75; diminishing beyond ~45 — [2506.07636](https://arxiv.org/html/2506.07636)

### Inferences
- Realistic target for us: 2k–5k re-mapped trajectories on Gemma 4 12B/31B with LoRA should land in the 15–25% regime on public-repo tasks if the schema mapping is clean; the 40%+ results all needed >14k trajectories, full FT and 80–128k context, none of which we have.
- Loop suppression is the cheapest post-SFT win for a 6-minute budget: either a harness-level repeated-action guard in the ADK config or a few hundred RL/DPO steps.

### Gaps
- No published Gemma 3/4 SWE-agent fine-tune with SWE-bench numbers was found (search returned only Gemma-3-270M code tutorials).
- No data-scaling curve exists for LoRA rank 64–128 on 8B–32B models; all 30%+ results are full FT.
- No study of int4 QAT checkpoints as LoRA bases for agentic SFT.

---

## (4) Training on compacted / summarized / truncated histories

### Takeaway
LLM-summary compaction is lossy (retains ≤16% of recurring details and damages 5–64% of otherwise-completed repairs), but trained or carefully-timed compaction helps small models under tight windows (+9.2 V for AutoCompact; +12 points on long sessions for a 27B at 49k). Training-time approaches that worked: truncation-ratio sampling (SERA), latent/compressed observations with text-anchored recent turns (ACD), learned memory tools (SWE-MeM), and summary-based managers (SWE-Master, −42% input tokens).

### Cited Findings
- Compact-Bench: across 14 summarizer models, compaction damages 5–64% of the 22 repairs full-history agents complete; LLM summaries keep ≤16.0% of recurring details vs 79.1% for mechanical extraction — [2610.31710](https://arxiv.org/abs/2610.31710)
- Bud ContextCompact 2.0: 27B model, 49k window, 100 Verified tasks: long-session resolve 41→53% (+12.3, CI +1.6..+23.0); gain vanished under a 45-min budget because each summary took median 161 s; an 8B model submitted after ~10 turns at 5–8k tokens and never needed compaction — [2610.25516](https://arxiv.org/abs/2610.25516)
- AutoCompact: SFT on judge-corrected compaction decisions, then RL → +9.2 V absolute; holds for both 256k and 16k windows — [2610.02163](https://arxiv.org/abs/2610.02163)
- ACD/LOHA: compress old tool observations to soft tokens, keep agent turns + last K=3 observations as text: Qwen3-4B 14.5→12.1 with 43% less context; under a hard 32k limit, 21.1% vs 11.1% for full-text; 1.9x throughput on one GPU — [2609.31430](https://arxiv.org/abs/2609.31430)
- SWE-MeM: memory tool + memory-aware GRPO: 4B 43.4, 30B 60.2 V — [2606.28434](https://arxiv.org/abs/2606.28434)
- SWE-Master: summary-based context manager cut MiniMax-M2.1 input tokens 42% (2.8M→1.6M), peak per-trajectory 55.4k→28.5k; ~20% of early RL trajectories truncated by budget, 24.3% of those would have resolved; forced-submit with 0.5 reward multiplier; zero-masking truncated trajectories (DeepSWE style) collapsed training — [2602.03411](https://arxiv.org/html/2602.03411)
- SERA: trajectories >32k truncated by a "truncation ratio" (fraction of steps that fit); training at 32k hurt 64k eval vs longer-context baselines — [2601.20789](https://arxiv.org/html/2601.20789)
- SWE-Dev: RoPE extension 32k→160k dropped 7B 22.8→21.8; 32B unaffected — [2506.07636](https://arxiv.org/html/2506.07636)
- Nebius: released trajectories with linear history and *no* truncation at max sequence length; avg 64.3 turns — [blog](https://nebius.com/blog/posts/openhands-trajectories-with-qwen3-coder-480b)

### Inferences
- With 6 min/task and an int4 model on L4s, we will be in the Bud "8B regime": short trajectories, so the bigger lever is training the model to finish in ≤20–30 turns rather than compaction. If we do compact, mechanical extraction (keep edits, failing-test output, file paths) beats LLM summaries, and training must see such compacted views.

### Gaps
- No paper trains *only* on summarized histories (without full-history data) and reports the result.
- No compaction study on ADK/YAML multi-agent configs.

---

## (5) Most promising ideas for us, ranked by expected gain per GPU-hour

### Takeaway
Reuse free trajectories, convert the schema, keep sequences short, and spend GPUs on SFT rather than environment construction.

1. **Re-map existing permissive trajectories into the 9-tool schema and SFT Gemma 4 12B/31B with LoRA.** Sources: nebius/SWE-rebench-openhands-trajectories (32,161 successful, CC-BY-4.0, native tool_calls), nvidia/SWE-Hero (34k, CC-BY-4.0), Nemotron-SWE-v1 (59k, CC-BY-4.0), SWE-smith-trajectories (MIT, `tool` split), allenai/Sera-4.6-Lite-T2 (25k, Apache-2.0). Zero API cost; the field's 8B results (SERA 31.7, Klear 39.0, SWE-Lego 42.2) all come from exactly this kind of data. Filter to ≤30 turns and ≤16–24k tokens to fit L4 memory and the 6-min budget (Bud: short sessions never needed compaction; SWE-Dev: 30 rounds already gives 34.0 of 36.6).
2. **Length-filtered curriculum + error masking** (SWE-Lego: +2.0 curriculum, +0.4–0.8 masking, heuristic trajectory-length curriculum beat a learned difficulty scorer). Costs nothing extra.
3. **Self-generated unverified/soft-verified trajectories from Gemma 4 31B on the 129 public tasks and procedural bugs**, no teacher API (SERA: verification threshold did not matter, p=.066; SWE-Dev RFT at 2.3k traj → 21.2 on 7B). Run generation in GPU sessions with a local SWE-smith-style harness; keep trajectories with a plausible patch (line-recall vs. gold ≥0.25 where gold exists).
4. **Teacher-free procedural bug injection on offline-cloned Python repos with tests** (SWE-smith procedural: 40.2% yield at 0 cost, downstream 8.6 vs 9.2 for the best LM method; SWE-Synth synthetic ≥ real at matched size). Also "revert the fix commit" of the 129 public tasks' upstream repos for extra tasks. Use for item 3's prompts and for a tiny held-out dev set on *unseen* repos to measure generalisation to private repos.
5. **Add Agentless-style single-turn skill data** (localization / repair / test-gen JSONL from nvidia Nemotron-Cascade-SFT-SWE, CC-BY-4.0) as a cheap mixture: Kimi-Dev shows these priors transfer to agent mode, and short sequences are cheap on L4s.
6. **Loop guard + early-submit policy in the ADK YAML config**, not in weights (SWE-Protégé: 31% of SFT-only 7B trajectories loop >10 steps; the loop heuristic alone was worth +4.4). Zero GPU cost.
7. **Inject synthetic code-graph tool usage into trajectories** (replace a `grep`/`find` call with get_code_neighbors / search_similar_code whose output is computed offline from the repo) so the model learns the three tools no public dataset covers. Speculative; needs a small ablation.
8. **Only if time remains: ~100–300 steps of DPO/GRPO against test outcomes on the 129 tasks** (SWE-Master 4B: SFT 27.6 → RL 33.4; SWE-Protégé RL −30 points of looping). RL with vLLM rollouts at int4 on L4x4 is slow; treat as optional.

Budget sanity check (inference, not sourced): 3k trajectories x ~15k tokens ≈ 45M tokens per epoch; at a few thousand tokens/s on L4x4 that is roughly 5–10 h per epoch for a 12B LoRA, so 2–3 epochs plus eval fits one account's weekly allotment. 31B at int4 roughly doubles this.

---

## (6) Not feasible for us and why

- **Frontier-teacher distillation at scale.** SWE-smith paid ≈$0.52 per Claude-3.7 trajectory ([SERA](https://arxiv.org/html/2601.20789)); 5k trajectories ≈ $2,600 vs our <$50 (≈100 trajectories). SERA's own cheap route still used GLM-4.6 self-hosted for 40 GPU-days.
- **Large-scale environment construction.** OpenSWE spent $891k on environments + $576k on trajectories ([2603.13023](https://arxiv.org/abs/2603.13023)); Skywork filtered 146,568 PR instances to 10,169 ([2506.19290](https://arxiv.org/html/2506.19290)); SWE-rebench V2 needed a setup agent + judge ensemble. We can only build dozens of envs offline.
- **Full fine-tuning / long-context SFT.** SWE-Lego trains 128k sequences with full FT (LR 1e-4, 4 epochs) and SWE-Master 80k; SERA 32B used 960 GPU-hours. L4x4 (96 GB) with int4 QAT base + LoRA realistically caps at ~16–32k sequences.
- **Multi-turn RL with execution feedback at 100–150 turns** (SWE-Master: 150 turns, 108k context, 5,400 s timeouts; DeepSWE/SkyRL-style setups). Session limit 12 h and no internet (no Docker image pulls) make this impractical beyond toy scale.
- **Test-time scaling with verifiers** (R2E-Gym 34.4→51 with hybrid verifiers; SWE-Lego TTS@16 +7.4). 6 min/task leaves room for pass@1 only.
- **Expert-in-the-loop (SWE-Protégé 42.4 with Opus 4.1 at ~4 calls/task).** The eval harness has no external model; the in-house 7B/32B experts only gave 17.0 / 20.8.
- **LLM-summary compaction at inference.** Median 161 s per summary on a 27B ([2610.25516](https://arxiv.org/abs/2610.25516)) is incompatible with a 6-min task budget on L4s; mechanical truncation is the only option.
- **Gated / unclear-license data.** OpenSWE is gated (legal name + organisation) and README says AGPL-3.0; SWE-Lego, Skywork and SWE-Master trajectories were not confirmed released; SWE-Swiss data could not be located.

### Gaps (cross-cutting)
- No source gives SWE-bench results for Gemma 4 at any size, nor for int4-QAT bases with LoRA, so all absolute targets above are extrapolated from Qwen results.
- Hidden-test repos are private; no paper measures synthetic-task training on repos absent from pretraining, except SERA's caveat that its specialisation gains "may be biased" by Django/Sympy/Sphinx being public.
