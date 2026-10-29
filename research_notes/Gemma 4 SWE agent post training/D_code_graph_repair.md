# Cluster D: Code-graph and repository-structure reasoning for localization/repair, and training LLMs to use graph tools (2024–Oct 2026)

Scope note: arXiv IDs are cited inline as [arXiv:ID](https://arxiv.org/abs/ID). Benchmarks: SWE-bench Lite (300 tasks), SWE-bench Verified (500 tasks). "Func Acc@k" = all gold functions in top-k (LocAgent convention); "R@k" = recall at k; "F1" = CodeScout convention (variable-length prediction). Numbers are not directly comparable across conventions.

## (1) Paper table: graph type, consumption mode, localization metrics, resolve gains, model sizes

### Takeaway
Graph-guided localization reliably improves function-level localization and gives +2 to +6 pp end-to-end resolve gains when plugged into fixed repair pipelines, but gains are largest when the graph is exposed through 2–3 compact tools (name lookup + typed neighborhood traversal) rather than a query language, and the only systems that make small (7B) models competitive with frontier models on localization are ones that *train* the model on tool trajectories.

### Cited Findings

| System | Graph construction | How LLM consumes it | Localization | Resolve-rate effect | Models tested |
|---|---|---|---|---|---|
| CodexGraph [arXiv:2408.03910](https://arxiv.org/abs/2408.03910) | Neo4j DB; MODULE/CLASS/FUNCTION/FIELD nodes; CONTAINS/INHERITS/HAS_METHOD edges, two-phase AST indexing | Agent writes NL query, second LLM translates to Cypher ("write then translate"), iterative | not reported separately | SWE-bench Lite (257 tasks, Sympy excluded for OOM): GPT-4o 22.96%, DeepSeek-Coder-V2 12.06%, Qwen2-72B 1.95% | GPT-4o, DS-Coder-V2, Qwen2-72B |
| RepoGraph [arXiv:2410.14684](https://arxiv.org/abs/2410.14684) | tree-sitter; **line-level** def/ref nodes; invoke/contain edges; builtins and third-party refs filtered | k-hop ego-graph flattened into prompt (procedural) or `search_repograph` action (agent) | not reported | Lite: Agentless GPT-4o 27.33 to 29.67; SWE-agent GPT-4o 18.33 to 20.33; AutoCodeRover GPT-4 19.00 to 21.33; RAG GPT-4 2.67 to 5.33; "32.8% average relative" | GPT-4/4o, Claude-3.5 (appendix) |
| LocAgent [arXiv:2503.09089](https://arxiv.org/abs/2503.09089) | Heterogeneous directed graph: directory/file/class/function; contain/import/invoke/inherit; BM25 inverted index over entity ids | 3 tools: SearchEntity, TraverseGraph (type-aware BFS, hops, direction, relation filter; tree-formatted output), RetrieveEntity | Lite: file Acc@5 94.16 / func Acc@10 77.37 (Claude-3.5); Qwen2.5-7B-ft 88.32/71.53; Qwen2.5-32B-ft 92.70/77.01; BM25 61.68/36.86; CodeRankEmbed 84.67/58.76 | Agentless editing (Claude-3.5): Pass@1 26.31 to 27.92 (Claude loc) / 26.79 (Qwen-32B-ft loc); Pass@10 33.58 to 37.59 | Claude-3.5, GPT-4o, Qwen2.5-Coder-7B/32B (LoRA SFT) |
| CoSIL [arXiv:2503.22424](https://arxiv.org/abs/2503.22424) | Dynamically built module call graph then function call graph; no index | Iterative LLM-driven search with pruner + reflection | Top-1 func 43.3% Lite / 44.6% Verified (Qwen2.5-Coder-32B) | +2.98%–30.5% when plugged into Agentless | Qwen2.5-Coder-32B |
| OrcaLoca [arXiv:2502.00350](https://arxiv.org/abs/2502.00350) | Simplified code graph + priority scheduling, relevance scoring, distance-aware pruning | Agent tools | Function match 65.33% Lite | +6.33 pp resolve for an open-source framework | (closed models) |
| KGCompass [arXiv:2503.21710](https://arxiv.org/abs/2503.21710) | KG linking issues/PRs to file/class/function entities | Narrow to 20 candidate functions; KG paths injected into repair prompt | Func-level 56.0% Lite; 89.7% of localized bugs lacked explicit hints, found via multi-hop traversal | 58.3% Lite single-model (Claude-4 Sonnet), $0.2/repair; relative lift +50.8% Claude-4, +30.2% Claude-3.5, +115.7% DeepSeek-V3, +156.4% Qwen2.5-Max | Claude-4/3.5, DeepSeek-V3, Qwen2.5-Max |
| CGM [arXiv:2505.16901](https://arxiv.org/abs/2505.16901) | Repo code graph | Graph injected into **attention** via adapter + agentless graph RAG | — | 43.00% Lite, Qwen2.5-72B | 72B |
| LingmaAgent/RepoUnderstander [arXiv:2406.01422](https://arxiv.org/abs/2406.01422) | Top-down repo KG | MCTS exploration | — | +18.5% relative vs SWE-agent on Lite | GPT-4-class |
| SWE-Search [arXiv:2410.20285](https://arxiv.org/abs/2410.20285) | No code graph; MCTS over agent action tree with LLM value function | — | — | +23% relative across 5 models | mixed |
| Aider repo-map ([docs](https://aider.chat/docs/repomap.html)) | Graph ranking over file-dependency graph; most-referenced identifiers | Prompt-stuffed, default 1k-token budget, sent with every request | none reported | none reported | — |
| ARISE [arXiv:2605.03117](https://arxiv.org/abs/2605.03117) | Multi-granularity graph down to statement nodes with intra-procedural def-use edges | 3-tier tool API; central tool = data-flow slicing | Func R@1 0.43 to 0.60; line R@1 0.26 to 0.41 | SWE-agent + Qwen2.5-Coder-32B on Lite: 17.3 to 22.0% (+4.7 pp) | Qwen2.5-Coder-32B |
| DUALVIEW [arXiv:2607.01929](https://arxiv.org/abs/2607.01929) | Module-coupling, call, class-hierarchy, program-dependence graphs | Queryable interface returning text + rendered images | — | "consistent" gains on Pro/Verified (no numbers in abstract) | several |
| SHERLOC [arXiv:2606.24820](https://arxiv.org/abs/2606.24820) | Compact repo tools, no training | Hypothesis-driven reasoning LLM | 84.33% acc@1 Lite; 81.27% recall@1 Verified; ~30B matches agentic methods | +5.95 pp avg when locations+diagnosis injected into repair agents | multiple scales |
| SWE-Debate [arXiv:2507.23348](https://arxiv.org/abs/2507.23348) | Fault-propagation traces over dependency graph | Multi-agent debate then MCTS patcher | 81.67% file-level Lite (DeepSeek-V3-0324; per [search snippet](https://arxiv.org/pdf/2507.23348)) | SOTA among open frameworks (abstract) | DeepSeek-V3 |
| LARGER [arXiv:2605.16352](https://arxiv.org/abs/2605.16352) | Graph anchors aligned to lexical hits; confidence-filtered local expansion inside existing grep loop | No separate graph tool | +13.9 file Acc@5 on LocBench (tuned), +11.8 fixed | — | CLI agents |
| CodeAnchor [arXiv:2606.26979](https://arxiv.org/abs/2606.26979) | Lightweight call/inheritance topology injected as plain-text comments | Prompt-level anchors (Codex baseline) | +2.2 pp Func@5; −1.6 rounds; run-to-run variance roughly halved | Pass@1 +3.4 pp on medium repos; ~10% more input tokens | Codex |
| Thousand-Graph [arXiv:2608.26602](https://arxiv.org/abs/2608.26602) | Entity-only two-layer index, **no edges** | Prompt index | — | 92.1/94.2/95.6% Verified (base/1-layer/2-layer) with DeepSeek-V4-Flash | frontier |
| Prometheus [arXiv:2507.19942](https://arxiv.org/abs/2507.19942) | Unified KG + working memory | Multi-agent | — | 74.4% Verified with GPT-5 | GPT-5 |
| SpIDER [arXiv:2512.16956](https://arxiv.org/abs/2512.16956) | Syntax-tree graph built on demand | Dense retrieval seeds then graph candidate expansion with edge-type reason | Recall@20 +13% rel (contain edges), +27% rel (call edges) across 4 languages | — | retrievers |
| RepoNavigator [arXiv:2512.20957](https://arxiv.org/abs/2512.20957) | Language-server `jump`-to-definition (single tool) | RL (GRPO) from base model, no distillation | Verified file/func F1: 7B 51.63/27.49, 32B 67.75/34.09 (as tabulated in CodeScout) | — | Qwen2.5 7B/14B/32B |
| CodeScout [arXiv:2603.17829](https://arxiv.org/abs/2603.17829) | **No graph**; bash terminal only (ripgrep) | RL (GSPO) with F1 reward | Verified file/func F1: Qwen3-4B-Instruct base 49.73/13.27 to CodeScout-4B 68.52/36.78; 14B 68.57/40.32; 1.7B (RFT from 14B + RL) 55.46/28.22 | — | Qwen3 1.7B/4B/14B |
| ToolTrain/RepoSearcher [arXiv:2508.03012](https://arxiv.org/abs/2508.03012) | 5 name-based retrieval tools (GetRepoStructure, GetImportOfFile, SearchClass/Function/ClassMethod) | Rejection-sampled SFT + RL (nDCG@5 reward) | Verified func R@5: Qwen2.5-Coder-7B 25.82 to 62.38; 32B 48.19 to 68.55 (Claude-3.7 in same scaffold 66.08) | Agentless patching: 7B loc+7B patch 14.00% vs Agentless 7.60%; 32B 31.60% vs 25.80% | Qwen2.5-Coder-7B/32B |
| Small-agents study [arXiv:2610.33958](https://arxiv.org/abs/2610.33958) | LocAgent-style graph (file/class/func; contain/import/invoke/inherit), 26k nodes avg | Tools: grep/read/list, optionally `semantic_search` (Qwen3-Embedding-0.6B), optionally `find_entity` + `graph_neighbors` (raw edge lines, 50-line cap, 4k-char budget) | See section 2 | Fixed Qwen3-8B repair: 6/13/12/13 of 160 tasks resolved for grep/E/T/ET vs 26 with oracle locations | Qwen3 0.6B–32B, Gemma 4 E2B/E4B/12B |

- CodexGraph's authors note the approach is "comparable to AutoCodeRover" on Lite with GPT-4o and collapses with weaker backbones (Qwen2-72B: 1.95%) — [arXiv:2408.03910](https://arxiv.org/abs/2408.03910).
- LocAgent ablation (Qwen2.5-7B-ft): removing TraverseGraph drops func Acc@10 71.53 to 66.06; fixing hops=1 gives 66.79; keeping only `contain` edges gives 66.42; removing SearchEntity gives 53.28; removing BM25 index gives 60.22. Tree-formatted subgraph output beat row/incident/DOT/JSON formats on 37 hard tasks — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089).
- Code-completion graph work (GraphCoder [arXiv:2406.07003](https://arxiv.org/abs/2406.07003) control/data-dependence graph, +6.06 EM; RepoHyper [arXiv:2403.06095](https://arxiv.org/abs/2403.06095) semantic graph + link prediction; CoCoMIC [arXiv:2212.10007](https://arxiv.org/abs/2212.10007) cross-file context finder, +33.94% rel EM; GraphCodeAgent [arXiv:2504.10046](https://arxiv.org/abs/2504.10046) dual requirement/structure graphs on DevEval/CoderEval) targets completion/generation, not issue repair.

### Inferences
- Across frameworks, the realized end-to-end gain from a graph plug-in on a fixed repair stage clusters at +2 to +6 pp absolute (RepoGraph, OrcaLoca, ARISE, SHERLOC); larger relative lifts (KGCompass, CoSIL) come from weak baselines or candidate pruning to ~20 functions.
- The consumption modes that work for open/smaller models are compact tool outputs with exact entity ids (LocAgent, RepoSearcher, ARISE); Cypher-style query languages and prompt-stuffed line-level ego-graphs were only validated with GPT-4-class models.

### Gaps
- No paper reports localization or resolve numbers for Gemma-family models other than [arXiv:2610.33958](https://arxiv.org/abs/2610.33958) (localization only). DUALVIEW and SWE-Debate numbers were not verifiable beyond abstracts/snippets.

## (2) Small-model-specific evidence: do small models benefit more or less from graph tools?

### Takeaway
The only controlled scale study (0.6B–32B, 79k episodes, incl. Gemma 4 E2B/E4B) finds structured tools help small models *more*: the gain over grep-only peaks at +17–21 func-Acc@10 points at 1.7–4B and falls to +4–5 at 32B, but the gain is almost entirely exact-name lookup (hop 0) plus one neighborhood expansion (hop 1); small agents never chain traversals, and no tool flattens the depth gradient.

### Cited Findings
- Gemma 4 E4B on the 160-task depth-balanced Verified subset: grep-only 9.6% func Acc@10 vs 21.8 (E: +embedding search) / 21.8 (T: +graph tools) / 23.7 (ET) (p at most 5.5e-4); three-run means 13.5 vs 25.2 / 22.4 / 27.1. On all 500 tasks: 16.7 vs 28.5 / 26.8 / 29.7; scale-curve rerun 18.0 vs 28.2 / 29.1 / 32.4. Gemma 4 E2B: 4.3 vs 26.0 / 24.8 / 33.0. Gemma 4 12B greedy run was degenerate (51% hit step cap) and is not interpretable — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- E4B's file Acc@5 was already 53.8% with grep and did not change with tools; tools improved the *function named inside an already-found file* — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- Scale curve (Qwen3, strict func Acc@10, best structured arm minus grep): +9 at 0.6B, +21 at 1.7B, +19 at 4B, +15 at 8B, +14 at 14B, +5 at 32B; GEE odds-ratio 0.77–0.82 per parameter doubling (p<1e-40). From 8B up about half the strict gain is submission *form* (exact ids) rather than discovery; canonical rescoring cuts ET minus G from +15.0 to +8.7 (8B) and +4.4 to +2.2 (32B) — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- The 14B-to-32B fall is a fall in tool uptake (find_entity called in 96% of 14B T-episodes vs 45% at 32B), not grep competence; `graph_neighbors` is abandoned first (88% of T episodes at 8B, 13% at 14B) — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- Each extra hop between issue-mentioned code and gold-patched code multiplies odds of correct localization by 0.544 (Gemma 4 E4B), 0.47 (Qwen3-4B/8B), 0.55 (Qwen3-32B); hop-by-T interaction never significant, hop-by-E interaction >1 (embedding search degrades least with depth). In T episodes, `find_entity` first reveals gold in 46–53% of hop-0 episodes, `graph_neighbors` in 40–49% of hop-1; only 12–14% of episodes chain expansions; 64–72% of 2-hop requests were truncated by the 4k-char budget — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- Checkpoint matters as much as size: Qwen3-4B-Instruct-2507 gains nothing from embedding search (−1.1) and +8.1 from graph tools; the hybrid Qwen3-4B checkpoint gains +17.6 from embedding search. Forcing tool use did not help; raising budget from 30 to 50 steps did not rescue grep-only looping — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- SWE-bench Verified hop distribution: 186/128/78/76 tasks at hop 0/1/2/3+ (37%/26%/16%/15%); mean depth 1.17 with all edge types; calls are the edge type that connects most mentions to targets — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- LocAgent (larger models) reported graph agents degrade *least* with hop distance; 2610.33958 attributes the discrepancy to model scale and to LocAgent's compressed tree-formatted subgraph output vs raw edge lines — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089), [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- Untrained 7B models with graph tools are weak: LocAgent with off-the-shelf Qwen-7B scores func R@5 17.72 on Verified vs 34.12 for its fine-tuned CL-7B and 62.38 for ToolTrain-7B — [arXiv:2508.03012](https://arxiv.org/abs/2508.03012).
- Untrained Qwen3-1.7B scores ~0 F1 with bash-only localization; Qwen3-4B-Instruct 13.27 func F1 rises to 36.78 after RL — [arXiv:2603.17829](https://arxiv.org/abs/2603.17829).

### Inferences
- For Gemma 4 E4B specifically, the prior evidence predicts a large benefit from `get_code_neighbors`-style tools untrained (+10–14 pts func Acc@10) but concentrated on tasks whose issue names the target or its neighbor; deeper tasks (~30% of Verified) will remain failed unless the model is trained to chain traversals.
- Because file-level accuracy was unaffected for E4B, the marginal value of graph tools is function-level precision and exact-id form, which is precisely what an edit tool needs.

### Gaps
- 2610.33958 is a single-author preprint (posted 2026-10-25), not yet peer-reviewed; its data/traces are released. No independent replication on Gemma 4 found.
- No study measures graph-tool benefit for small models *after* tool-use training (all small-model training work uses name-lookup tools or bash, not typed-neighborhood tools).

## (3) Teaching models to use graph tools: methods and data recipes

### Takeaway
Three recipes exist: (a) rejection-sampled SFT on teacher trajectories (LocAgent: 768 trajectories, LoRA), (b) SFT warm-up + RL with a ranking reward on gold-patch-derived targets (ToolTrain: 5k SFT + 18k RL, +36 pts func R@5 at 7B), and (c) pure RL from a base model with F1/jump rewards (RepoNavigator, CodeScout: 4B trained on 1.6k instances x 8 rollouts in 200 steps). Localization-only RL needs no execution sandbox. Gold patches are increasingly used as *privileged* evidence to synthesize stage-specific supervision (P2T, SWE-Ladder) rather than only as a filter.

### Cited Findings
- LocAgent fine-tuning: 433 successful Claude-3.5 trajectories on the SWE-bench train split + 335 self-generated by the fine-tuned Qwen2.5-32B; total 768 samples; LoRA, 5 epochs, lr 2e-4, 128k max tokens; distilled to 7B. Cost: $0.05/example for 7B-ft vs $0.66 for Claude-3.5 — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089).
- ToolTrain: 28k (issue, PR) pairs from 600 repos; 10k sampled with Claude-3.7-Sonnet, filtered to 5k correct trajectories for SFT (3 epochs, batch 128); remaining 18k for RL (verl + SGLang, 1 epoch, temp 1.0, 12k prompt / 20k response). Reward = nDCG@k against gold-patch functions. Ablation at 7B file R@5: base 57.73, RL-only 73.18, SFT-only 79.31, SFT+RL 83.11. Tool-call success rate rose to >95% during RL even though reward was outcome-only. Trained model transferred to the CoSIL scaffold (func R@5 30.47 to 50.87) — [arXiv:2508.03012](https://arxiv.org/abs/2508.03012).
- CodeScout: ground truth from gold patches at file/module/function (LocAgent patch scripts); reward = sum of three F1s (+ binary on-time-termination term for 14B); GSPO without KL, Dr.GRPO advantage; loss masked on tool observations and on rollouts that never call the finish tool. CodeScout-4B: 200 steps, 1.6k instances, batch 8, 8 rollouts, 40k context, 6 turns max, 8xH100. 1.7B needed RFT warm-start from 14B rollouts (4k perfect trajectories) before RL. Precision is argued more important than recall for downstream repair — [arXiv:2603.17829](https://arxiv.org/abs/2603.17829).
- RepoNavigator: single `jump` tool via language server; GRPO from pretrained model; 7B outperforms 14B baselines, 32B exceeds GPT-5 on most metrics — [arXiv:2512.20957](https://arxiv.org/abs/2512.20957).
- P2T (Patches-to-Trajectories): distills the reference patch into a latent process graph of contextual facts and milestones, then scores blinded teacher continuations per step for progress and keeps shortest effective segments; 1.8k curated SWE-Gym instances give up to +10.8 Pass@1 on Verified over outcome-filtered SFT — [arXiv:2605.21996](https://arxiv.org/abs/2605.21996).
- SWE-Ladder: stage-specific privileged evidence from the gold patch (changed files/lines for localization; fail-to-pass tests for reproduction; diff for patching); router assigns each rollout step to a stage; frozen teacher sees only matching evidence; GRPO on full trajectory. Qwen3-4B-Instruct-2507 on Verified: 9.6 to 16.4 pass@1 (R2E-Gym; GRPO alone 10.4, full-patch OPSD 12.4); OpenHands 13.0 to 17.6 — [arXiv:2610.33667](https://arxiv.org/abs/2610.33667).
- SWE-Protégé: Qwen2.5-Coder-7B reaches 42.4% Verified by learning to call an expert sparingly (~4 calls/task) via SFT + agentic RL penalizing loops — [arXiv:2602.22124](https://arxiv.org/abs/2602.22124).
- LocAgent found off-the-shelf Qwen-7B "performs poorly due to its limited tool-use capability"; for models without native tool calling it put tool descriptions in the prompt — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089).

### Inferences
- No published work synthesizes graph-traversal trajectories *from the graph itself* (shortest path from issue-mentioned nodes to gold nodes) rather than from teacher rollouts; 2610.33958 explicitly lists "training the agent to chain expansions" as open future work.
- Localization RL is the cheapest RL in this space: string-level rewards, no Docker, short episodes (4–6 turns in CodeScout).

### Gaps
- No study of tool *selection* among heterogeneous retrieval tools (embedding vs graph vs grep) as a trained skill; CodeScout argues graph tools are unnecessary given RL, but did not test graph tools under the same RL recipe.
- No reported GPU-hour budgets below 8xH100 for localization RL.

## (4) Retrieval quality notes relevant to 256-d embedding search over node names

### Takeaway
Dense code retrievers beat BM25 by wide margins on issue localization (e.g., CodeRankEmbed func Acc@10 58.76 vs BM25 36.86 on Lite) but are a weak *standalone* localizer versus agents; in a small-model agent, embedding search adds +11–13 pts for 8B/E4B models (and +0 for one 4B checkpoint), its slope with hop depth is the flattest of any tool, and about half its benefit depends on retriever quality (a 256-d static embedder lost 5 of 13 points). No paper directly measures 256-d Matryoshka truncation on issue localization.

### Cited Findings
- Lite function Acc@10: BM25 36.86; E5-base-v2 51.09; Jina-Code-v2 52.19; CodeRankEmbed 58.76; LocAgent Claude 77.37 — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089). Embedding methods "perform poorly at the function level, even when the patched functions are explicitly mentioned in the query" because they treat the query as a whole — same source.
- WeakEmb experiment: swapping Qwen3-Embedding-0.6B for potion-base-8M (static, 256-d) lowers Qwen3-8B's E-arm func Acc@10 26.3 to 21.2 (p=0.077); fraction of `semantic_search` calls revealing gold halves (32% to 16%); the remaining gain came from ending grep loops — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958). Index text there = node id + first 40 lines of definition.
- SWE-Explore: sparse retrievers and a static-embedding retriever "remain close to Random on most metrics" while agentic explorers form a tier above classical retrieval — [arXiv:2606.07297](https://arxiv.org/abs/2606.07297) (quote via [arXiv:2610.33958](https://arxiv.org/abs/2610.33958)).
- CORE-Bench: "sharp drop from traditional code search to code retrieval in agentic coding settings"; simple SFT of embedders helps substantially — [arXiv:2606.11864](https://arxiv.org/abs/2606.11864).
- SpIDER: dense seeds expanded along contain/call edges raise Recall@20 by at least 13%/27% relative — [arXiv:2512.16956](https://arxiv.org/abs/2512.16956).
- Code representation matters: role-aware LLM summaries beat file-path representations by up to 40% Hit@5 with 10–21x smaller footprint; Agentless case study 94% Hit@6 — [arXiv:2607.11046](https://arxiv.org/abs/2607.11046).
- SweRank: SweRankEmbed-Large Acc@20 86.50 on Lite vs Gemini-Embedding 79.20 (per [search snippet](https://arxiv.org/pdf/2505.07849)); retrieve-and-rerank beats Claude-3.5 agents at far lower cost — [arXiv:2505.07849](https://arxiv.org/abs/2505.07849).
- Nemotron-CORTEXA: custom code embedder for file retrieval + localization agent; entity precision/recall 34.51/67.62 vs Agentless 17.54/58.37 on Verified; 68.2% resolve — [NVIDIA page](https://research.nvidia.com/labs/adlr/cortexa).
- Dimension truncation: jina-code-embeddings supports 64–896 dims and recommends 128/256-d for high-throughput indexing with full dims for reranking ([model page](https://jina.ai/models/jina-code-embeddings-0.5b/)); a 1C:Enterprise code-retrieval paper reports 256-d retaining 99.9% of 768-d nDCG@10 ([arXiv:2608.19957](https://arxiv.org/abs/2608.19957)); a fine-tuned EmbeddingGemma code-search card shows ~0.753 to 0.643 at 256-d ([HF card](https://huggingface.co/jasperan/embeddinggemma-code-search)).
- Hybrid note: in SWT-Bench, BM25 top-30 gets 87.7% recall at 3.4% precision; reranking with Qwen3-Embedding-0.6B reaches 73.0% recall, nearly identical to the 8B embedder (per [search snippet](https://arxiv.org/pdf/2601.13713)).

### Inferences
- With resolution against node *names* plus lexical fallback, our `search_similar_code` behaves like LocAgent's SearchEntity fuzzy tier; its best use is as an entry-point generator (hop-0 anchor) followed by `get_code_neighbors`, i.e., SpIDER-style seed-then-expand, rather than as the final localizer.
- 256-d is unlikely to be the bottleneck relative to embedder quality and index text choice; the LocAgent/CORE-Bench evidence suggests whole-issue queries are the bigger problem, so queries should be short keyword/entity phrases extracted by the model.

### Gaps
- No measurement of Matryoshka truncation on SWE-bench-style localization; no paper on name-resolved cosine search with lexical fallback as in our harness.

## (5) Most promising ideas for us (ranked by expected gain per GPU-hour)

### Takeaway
The highest-leverage, lowest-cost moves are harness-side canonicalization and a graph-path-grounded SFT dataset built from the 129 gold patches; localization-only RL is feasible but second priority; the paper-track novelty lies in graph-path-synthesized trajectories and hop-depth-stratified evaluation on the released graph dataset.

### Cited Findings (grounding for the ranking)
- Half the measured structured-tool gain at 8B and above is submission form; canonicalizing names recovers it without tools — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).
- Resolution is almost entirely conditional on correct localization (24–27% vs 1–2%); oracle locations roughly double a weak repair model's resolves (13 to 26 of 160) — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958). Oracle spans lift Agentless 28.0 to 40.3% and KGCompass 29.3 to 43.0% on Lite (DeepSeek-V3), with best-of-5 recovering ~87% of further headroom — [arXiv:2603.29067](https://arxiv.org/abs/2603.29067).
- SFT warm-up alone gave most of ToolTrain's gain (file R@5 57.73 to 79.31 vs 83.11 with RL) — [arXiv:2508.03012](https://arxiv.org/abs/2508.03012). LocAgent needed only 768 trajectories — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089).
- Gold-patch privileged supervision at 4B: +6.8 pass@1 (SWE-Ladder) — [arXiv:2610.33667](https://arxiv.org/abs/2610.33667).

### Inferences (the ranked ideas)
1. **Canonical-id discipline (near-zero GPU cost; not a paper contribution alone).** Prompt/SFT E4B to always cite `path.py:Class.func` ids returned by `get_code_neighbors`/`search_similar_code`, and have the harness-side wrapper canonicalize line numbers / `::` / bare names before `edit_file`. Expected: recovers the "form" half of the tool gain and raises valid-edit rate.
2. **Graph-path-grounded oracle trajectories (CPU synthesis + ~2–4 GPU-h LoRA SFT; novel).** For each of the 129 tasks, compute mentions (regex over issue), gold nodes (from patch hunks), shortest typed paths in the NetworkX graph, and render them as `search_similar_code` then `get_code_neighbors(edge_type)` then `read_file` then `edit` trajectories with brief rationales (teacher only for rationale text, within $50). This is the "train agents to chain expansions" gap named in [arXiv:2610.33958](https://arxiv.org/abs/2610.33958) and differs from P2T/SWE-Ladder (which never use a code graph). Oversample hop-2/3 paths to counter the depth gradient.
3. **Hop-depth-stratified evaluation on the competition graph (CPU only; paper contribution).** Label the 129 public tasks by deterministic hop depth using the released graph and report per-bucket Acc and resolve for every variant; this directly uses the released graph/embedding dataset the paper track rewards and gives a controlled difficulty axis.
4. **Checkpoint-specific tool ablation for Gemma 4 E4B (about 1–2 GPU-h inference).** Run G/E/T/ET arms on the 129 tasks before training; [arXiv:2610.33958](https://arxiv.org/abs/2610.33958) shows the same size can gain +17.6 or 0 from embedding search depending on checkpoint, so tool-set choice should be measured, not inherited.
5. **Compact 2-hop closure via `get_code_subgraph` (harness-side + SFT).** Small agents expand once and stop, and 2-hop outputs get truncated; teach the model to call `get_code_subgraph` on the top-k neighbors and present a ranked, deduplicated closure within the 5,000-char cap (suggestion (ii) of [arXiv:2610.33958](https://arxiv.org/abs/2610.33958)). Novel as a trained behavior.
6. **Localization-only GRPO with nDCG/F1 reward (about 10–20 GPU-h; feasible but second-tier).** No sandbox needed; 4–6 turn episodes; targets from gold patches of 129 tasks (augment with other public Python repos' PRs if allowed). CodeScout-4B used 1.6k instances x 8 rollouts; at L4x4 with int4 LoRA expect about an order of magnitude fewer steps, so run after SFT (ToolTrain shows SFT-then-RL ordering matters).
7. **Localization-first time split.** Spend the first ~1/3 of the 6-min budget on a localization sub-episode that returns at most 5 canonical nodes with source, then repair; the Beyond-Localization and 2610.33958 oracle numbers bound the upside at roughly 1.5–2x current resolve for weak repair models.
8. **Seed-then-expand retrieval (harness prompt only).** Use `search_similar_code` with short entity-like queries as seeds and expand along `calls`/`contains` (SpIDER [arXiv:2512.16956](https://arxiv.org/abs/2512.16956)); cheap, likely +recall at hop 1–2.

### Gaps
- No prior evidence on whether synthesized (non-teacher) graph trajectories transfer as well as teacher rollouts; this is the main risk of idea 2 and the reason to keep a small Claude/Gemini-rationale budget.

## (6) Not feasible and why

### Takeaway
Architecture-level graph integration, execution-based end-to-end RL, inference-time search, and query-language graph interfaces are out of reach under Kaggle L4x4, 6-min/task, and a 4B research vehicle.

### Cited Findings
- CGM integrates the graph into attention with an adapter and reached 43% only with Qwen2.5-72B — [arXiv:2505.16901](https://arxiv.org/abs/2505.16901).
- End-to-end RL requires Docker execution per rollout, which RepoNavigator calls "prohibitively expensive" and is why localization RL was adopted — [arXiv:2512.20957](https://arxiv.org/abs/2512.20957); CodeScout's *localization-only* 4B RL still used 8xH100 — [arXiv:2603.17829](https://arxiv.org/abs/2603.17829).
- CodexGraph's Cypher interface scored 1.95% on Lite with Qwen2-72B vs 22.96% with GPT-4o — [arXiv:2408.03910](https://arxiv.org/abs/2408.03910).
- ToolTrain sampled 10k Claude-3.7 trajectories for SFT — [arXiv:2508.03012](https://arxiv.org/abs/2508.03012); LocAgent's Claude trajectories cost $0.66 each — [arXiv:2503.09089](https://arxiv.org/abs/2503.09089).
- MCTS frameworks (SWE-Search, LingmaAgent, SWE-Debate's MCTS patcher) trade inference compute for accuracy — [arXiv:2410.20285](https://arxiv.org/abs/2410.20285), [arXiv:2406.01422](https://arxiv.org/abs/2406.01422), [arXiv:2507.23348](https://arxiv.org/abs/2507.23348).
- Forcing structured tool calls and enlarging step/context budgets did not help small agents — [arXiv:2610.33958](https://arxiv.org/abs/2610.33958).

### Inferences
- Not feasible: CGM-style graph-attention adapters (architecture change, 72B); DeepSWE-style execution-reward RL (sandbox cost on L4x4); MCTS/debate/best-of-10 at inference (6-min budget on 4B); Cypher/graph-DB query interfaces (collapse with non-frontier models); large teacher-distilled trajectory sets (over $50); DUALVIEW-style image rendering (harness tools return text only); training a new 256-d embedder (embeddings are precomputed by the harness); ARISE-style statement-level data-flow graphs (harness graph is function-granular).
- Marginal: full localization RL at CodeScout scale; recommend a scaled-down GRPO only after SFT shows headroom.

### Gaps
- Gemma 4 12B/31B tool-use behavior with graph tools is uncharacterized ([arXiv:2610.33958](https://arxiv.org/abs/2610.33958)'s 12B run was degenerate), so transfer of E4B findings to the final 12B/31B submission is an open risk.
