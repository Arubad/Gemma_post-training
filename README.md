# Gemma 4 Developer Agent: post-training and submission code

This repo contains everything needed to build, train, evaluate and package submissions for Kaggle's *Google - The Gemma 4 Developer Agent Competition*. It implements the recipe from `reports/Gemma 4 SWE agent post training.md`:

- **Harness-exact LoRA SFT** on a bf16 dequantization of the served `-qat-w4a16-ct` checkpoint.
- **Training data** from two sources: public trajectories re-mapped to the 9 harness tools, and **graph-path trajectories** synthesized from the released code graph (the paper contribution).
- **One round of rejection-sampling fine-tuning (RFT)** in the real harness.
- **Scorer-faithful local evaluation** with hop-stratified reporting.

The data → train → eval → package flow (training on E4B, plus a 31B prompt-only final):

```
HF trajectories ──remap──┐
code graph + gold ──synth─┼─► calibration (captured exact prompt/tools) ─► render+mask (model's own template)
RFT harvest (proxy logs)─┘                                                     │
                                                                               ▼
              w4a16-ct ──dequantize──► bf16 ──LoRA SFT (DDP x4 L4)──► fix/validate adapter
                                                                               │
              vLLM (scorer flags) ◄── proxy :8000 ◄── swegemma eval ◄──── submission dir ──► submission.zip
```

## Layout

| Path | What it is |
|---|---|
| `gemma_swe/` | Library. Pure-Python modules run anywhere; `dequant`, `losses` and `chat_render` need torch/transformers. |
| `scripts/` | CLIs used by the notebooks. Run any of them with `--help`. |
| `notebooks/nb00..nb06` | Kaggle notebooks (`.py` percent format plus generated `.ipynb`). |
| `profiles/*.yaml` | One profile per submission candidate: model, adapter, sampling, budgets, single agent or localizer variant. |
| `submission_template/prompts/` | System prompt, localizer prompt and localizer addendum. |
| `tests/` | 33 unit and integration tests. Run them with `pytest tests` (needs pyyaml, torch, safetensors, numpy). |

## Kaggle setup (one time)

1. Upload this repo as a private Dataset named **`gemma-swe-code`**.
2. Attach these inputs to every notebook:
   - the competition data;
   - the competition **wheelhouse** (the dataset containing `swegemma-*.whl`);
   - Kaggle Model **google/gemma-4**, in the needed `-it-qat-w4a16-ct` variants.
3. Paths are discovered automatically by `gemma_swe/kaggle_env.py`.
4. Datasets you will create from notebook outputs: `gemma-swe-wheels`, `gemma-swe-data`, `gemma-swe-adapters`.

## Runbook

GPU-h counts are L4x4 hours; Kaggle bills them at 2x quota. Split independent arms across the 2–3 team accounts.

| # | Notebook | HW / net | GPU-h | Produces |
|---|---|---|---|---|
| 0 | `nb00_prepare_data_internet` | CPU, internet on | 0 | `remapped.jsonl` (target 2k–5k), training wheels |
| 1 | `nb01_capture_and_synth_cpu` | CPU | 0 | `calibration.json`, task split, `hop_map.json`, `graph_synth.jsonl` |
| 2 | `nb02_verify_gpu` | L4x4 | ~1.5 | Pass/fail gates (below) and an untuned E4B baseline |
| 3 | `nb03_train_e4b` | L4x4 | 3–5 | `adapters/<run>` and a validated submission zip |
| 4 | `nb04_local_eval` | L4x4 | ~1/seed | Held-out (`rich`, 48 tasks) mean ± std per arm, per hop bucket |
| 5 | `nb05_rft_round` | L4x4 | 8–13 | `rft_round1.jsonl`, adapter `rft1` |
| 6 | `nb06_package_finals` | CPU | 0 | `submission_e4b_lora.zip`, `submission_31b_prompt.zip` |

**Timeline**

- **Paper (Nov 12):** NB00–NB04 with the `e4b_prompt` vs `e4b_lora` arms gives the first complete experiment.
- **Leaderboard (until Dec 2):** NB05, the ablations, and the 31B finals.

### Verification gates (NB02). Do not train until all of these pass.

1. The dequantized bf16 weights match the library's own int4 unpacker.
2. A **zero-init adapter gives identical outputs** to the base model in vLLM, and a random adapter gives different outputs. This proves the LoRA is actually applied, not silently ignored.
3. `verify_serving.py render`: our training render matches **vLLM's own `/tokenize`** token for token, for real records.
4. `verify_serving.py parser`: multi-line `edit_file` / heredoc arguments round-trip through the `gemma4` tool parser.

## Design decisions worth knowing

- **Tool-call turns carry no text.** Gemma 4's chat template renders an assistant message as *tool call → tool response → content*. It closes the turn with `<turn|>` whenever that content is non-empty. So a sentence the model writes before a call is re-rendered after the result, and the next prompt is left without a model-turn opener; the community reported this as "premature `<turn|>` → loops". Training data and the system prompt therefore use bare tool calls (`rationale_mode=none`). Rationales are kept in the records and can go into the thought channel instead (`profiles/e4b_lora_think.yaml`, an ablation). I confirmed this against the Gemma 4 template from the `unsloth/gemma-4-E4B-it` mirror. Gate 3 must confirm it on the served checkpoint.
- **Calibration is captured, not guessed.** `capture_harness_prompt.py` runs `swegemma eval` against a scripted mock model and records:
  - the literal system and user prompts;
  - the tool JSON;
  - whether tool results arrive as `{"result": "<json>"}`.

  Every record is rendered through this capture. Re-capture whenever `system.md` or the wheelhouse changes.
- **Masks come from the model's own template** (`chat_render.py`). For each step:
  - Labels are exactly the tokens the model emits at serve time, including the `<|tool_response>` and `<turn|>` stop tokens.
  - Generation-only glue, such as the empty thought channel that 12B/31B prefill, is prompt, not label.
  - Inconsistent histories are dropped, never mis-labelled.
- **Training base = dequantized served weights.** `dequant.py` turns int4 × group scale into bf16, so train equals serve. LoRA targets are explicit language-model `q/k/v/o/gate/up/down` names. A plain `q_proj` suffix would also hit the vision tower's `Gemma4ClippableLinear` and crash PEFT. `check_adapter.py --fix` renames keys to the served checkpoint's module paths and drops tensors vLLM cannot serve.
- **No unseen edits.** Re-mapped views are split into ≤150-line / ≤10k-character `read_file` chunks. Synthesized trajectories read every line they edit. Scratch scripts go to `/tmp` via heredoc, because `write_file` writes only into `/workspace`, where every file leaks into the patch.
- **Budgets:** 5 min, 40 calls (E4B) or 30 calls (31B), and 120 s command timeout. About 120 tasks run sequentially inside 12 h including sandbox setup, and an overrun errors the whole run.
- **Held-out split:** all of `Textualize/rich` is excluded from every training source, including the HF data. One public-LB task is 0.017 and identical zips have scored 0.10 vs 0.15, so decide only on local mean ± std over ≥3 seeds.

## Not verifiable offline (check in NB01/NB02)

- **`swegemma eval` flags.** I assumed `--tasks --snapshots-dir --submission-dir --results-dir --sandbox --concurrency --max-tool-calls --max-time-minutes --timeout-seconds --task-ids` from the README. NB01 prints `--help`; pass extra flags (e.g. graphs dir) via `--extra`.
- **Endpoint discovery.** The CLI is assumed to attach to an already-running server on `127.0.0.1:8000`, which is where the proxy listens. If it routes adapters itself, `adapter_used` in `local_eval.py` output shows the adapter name; if not, use `--route-base-to main_lora`.
- **`task_results.jsonl` field names.** `harvest.py` accepts `instance_id/task_id/id` and `resolved/is_resolved/success/score`.
- **Whether ADK echoes `reasoning_content` back in history.** This only matters for the thinking ablation, and RFT proxy logs will show it.
- **31B LoRA.** `train_sft.py --strategy device_map` exists but is unproven on 4×L4: the bf16 weights are ~62 GB, and it needs `max-len` ≤ 8k. The default 31B final is prompt-only.

## Paper ablation arms (one account each)

| Arm | How |
|---|---|
| Prompt-only control | `profiles/e4b_prompt.yaml` |
| +HF re-mapped SFT | NB03 with `SOURCES=[remapped]` |
| +graph-path SFT | NB03 with `SOURCES=[remapped, graph_synth:2.0]` |
| +RFT round | NB05 |
| Rationale in the thought channel | `profiles/e4b_lora_think.yaml` (build data with `--rationale-mode thinking`) |
| Localizer sub-agent | `profiles/e4b_localizer.yaml` |
| Scale transfer | `profiles/31b_prompt.yaml` vs `31b_lora.yaml` |

Report each arm with `eval_report.py`: rate mean ± std, per repo, per hop bucket (0 / 1 / 2 / 3+ / unreachable), submit rate, loop rate, and calls per task.

## Licences

The training data comes from `nebius/SWE-rebench-openhands-trajectories` and `nvidia/SWE-Hero-openhands-trajectories` (CC-BY-4.0) and `SWE-bench/SWE-smith-trajectories` (MIT). Attribute them in the writeup.
