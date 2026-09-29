# Data Card

The `data/` folder contains raw JSONL model-run records only. Summary tables
aggregated from these records, sample manifests, and documentation live under
`code/`. Every record comes from a recorded model call. Hit-based proxy scores
and the intervention fields of the ToolSandbox shared-state control files are
documented below.
Most raw JSONL records are unique at the `(instance_id, step_t)` level. A few
diagnostic files contain repeated `(instance_id, step_t)` rows because they
include repeated protocol probes or multi-seed sweeps; in those cases, use
`seed` and/or the paper slice manifest under
`code/slice_manifests/paper_slices.json` to recover the exact paper-facing
slice.

## Folder layout

```
data/
├── bfcl/               BFCL-v4 (function-calling, high local sufficiency)
├── contextbench/       ContextBench (context-heavy code retrieval)
├── openai_budget_sweep/OpenAI receiver-side B-sweep raw outputs
├── repobench/          RepoBench (cross-file completion)
├── swe_bench_lite/     SWE-bench Lite (issue-resolution, capability transfer)
├── tau_bench/          τ-bench (customer-service tool use, boundary regime)
└── toolsandbox/        ToolSandbox (stateful tool execution, noise regime)
```

## File-naming convention

Within each benchmark folder, filenames encode the protocol:

| Filename pattern                          | Protocol                                                     |
|-------------------------------------------|--------------------------------------------------------------|
| `qwen-receiver-only.jsonl`                | RO — receiver acts on local view only                        |
| `qwen-single-full.jsonl`                  | SF — single agent reads the full broader context             |
| `qwen-coder-single-full.jsonl`            | SF using Qwen3-Coder (SWE-bench Lite only)                   |
| `qwen-80B-single-full.jsonl`              | SF using Qwen3-Next-80B (ToolSandbox stronger-receiver ctrl) |
| `qwen-ma-9B-from-9B.jsonl`                | MA matched: Qwen3.5-9B sender → Qwen3.5-9B receiver          |
| `qwen-ma-9B-from-80B.jsonl`               | MA main: Qwen3-Next-80B sender → Qwen3.5-9B receiver         |
| `qwen-ma-9B-from-coder.jsonl`             | MA "main" for SWE-bench Lite: Qwen3-Coder sender             |
| `qwen-ma-80B-from-80B.jsonl`              | MA self-pair on Qwen3-Next-80B (ToolSandbox control)         |
| `qwen-ma-strong-receiver.jsonl`           | MA into a stronger receiver (control)                        |
| `qwen-ma-canonical-json-format.jsonl`     | MA with canonical-JSON message encoding (ToolSandbox knob)   |
| `qwen-ma-decision-header-json-format.jsonl`| MA with decision-header-JSON encoding (ToolSandbox knob)    |
| `qwen-ma-prose-format.jsonl`              | MA with prose message encoding (ToolSandbox knob)            |
| `qwen-ma-perturbed-control.jsonl`         | MA with a step-0 wrong-artifact perturbation (shared-state control; see *Intervention rows*) |
| `qwen-ma-repaired-control.jsonl`          | Same perturbation plus a targeted step-1 repair (see *Intervention rows*) |
| `selector-adaptive.jsonl`                      | Adaptive selector (`agent_type=selector_adaptive`)                   |
| `selector-always-send.jsonl`                   | Static baseline `selector_always_send`: always forward sender message |
| `selector-always-skip.jsonl`                   | Static baseline `selector_always_skip`: always suppress sender message |
| `selector-always-rewrite.jsonl`                | Static baseline `selector_always_rewrite`: always rewrite to canonical-JSON |
| `recoding-probe-strict.jsonl`             | Strict-recoding probe of the representation-sensitivity claim |
| `openai_budget_sweep/pilot_raw.jsonl`     | Initial OpenAI receiver-side budget-sweep raw outputs |
| `openai_budget_sweep/hard_raw.jsonl`      | Main hard-sample OpenAI receiver-side budget-sweep raw outputs |

## Schema (JSONL records)

Every record in the 42 Qwen benchmark files (all files except
`recoding-probe-strict.jsonl` and `openai_budget_sweep/`) is a single dict with at least:

| Key | Type | Meaning |
|---|---|---|
| `run_id` | str | Originating run identifier |
| `benchmark` | str | One of `bfcl_v4` / `contextbench` / `repobench` / `swe_bench_lite` / `tau_bench` / `toolsandbox` |
| `agent_type` | str | Protocol identifier (matches filename pattern) |
| `sender_model` | str | Sender model id (empty for RO/SF) |
| `receiver_model` | str | Receiver model id (empty for SF) |
| `single_model` | str | Single-agent model id (only for SF) |
| `budget_tokens` | int | Sender message budget, in estimated tokens (see `message_tokens`) |
| `encoding_format` | str | Message format (e.g. `default`, `canonical_json`, `prose`) |
| `seed` | int | RNG seed |
| `instance_id` | str | Benchmark instance identifier |
| `step_t` | int | Step index within the trajectory |
| `total_steps` | int | Total step count for this instance |
| `message_raw` | str | Raw sender message (empty when no message protocol) |
| `message_tokens` | int | Estimated message length, `max(1, floor(len(message_raw) / 3.5))`; the budget enforcer truncates messages to `3.5 × budget_tokens` characters |
| `message_truncated` | bool | Whether the budget enforcer truncated the message |
| `receiver_input` | str | Final prompt seen by the receiver |
| `receiver_input_tokens` | int | |
| `receiver_output` | str | Receiver's raw output |
| `receiver_output_tokens` | int | |
| `action_type_pred` | str | Predicted next-action type (parsed) |
| `action_type_gold` | str | Gold next-action type |
| `artifact_pred` | str | Predicted target artifact (parsed) |
| `hit_artifact` | 0/1 | 1 iff `artifact_pred` matches gold |
| `hit_action` | 0/1 | 1 iff `action_type_pred` matches gold |
| `hit_joint` | 0/1 | 1 iff both `hit_artifact` and `hit_action` |

Unused fields are present as empty strings/zeros for schema uniformity.

The candidate list shown to the receiver (the `VALID ... values` lines in
`receiver_input`, `candidate_set_size = 192`) contains the benchmark's
candidates plus distractors added by the prompt constructor, including
`_negative_NN` padding labels.

### Hit-based proxy scores

Every Qwen record has `scoring_mode = "proxy"` and `scoring_latency_ms = 0`.
In these records `surprisal_bits`, `surprisal_no_msg`, and `csi_sample` are
hit-based proxy scores, not log-probabilities: `surprisal_bits` is 0.5 for a
joint hit, 3.0 for an action-only or artifact-only hit, and 6.0 otherwise;
`surprisal_no_msg` applies the same mapping to a receiver call without the
message; and `csi_sample = surprisal_no_msg - surprisal_bits`
(`code/selector/agents/ma_sequential.py`).

### Intervention rows

`toolsandbox/qwen-ma-perturbed-control.jsonl` and
`toolsandbox/qwen-ma-repaired-control.jsonl` mark intervened steps in
`intervention`: `perturb_wrong_artifact` for the 16 step-0 rows with
`is_perturbed = true` (`perturbation_type = "wrong_artifact_step0"`), and
`targeted_repair` for the 16 step-1 rows with `repair_mode = "targeted"` in the
repaired file. In every row, `action_type_pred`, `artifact_pred`, and the
`hit_*` fields are parsed and scored from the receiver's actual
`receiver_output`, using the same parser and exact-match rule as all other
records. The values the harness applied at an intervened step (the injected
wrong artifact or the gold repair) are kept separately in
`intervention_action_type_pred`, `intervention_artifact_pred`, and
`intervention_hit_*`; these fields are null in rows without an intervention.

## Schema (`data/openai_budget_sweep/`)

The appendix budget-sweep raw outputs are stored separately from
benchmark-native records because they reuse released Qwen sender messages
while varying only the receiver-facing message budget for OpenAI receivers.

| File | Meaning |
|---|---|
| `openai_budget_sweep/pilot_raw.jsonl` | 120 successful OpenAI receiver calls from the initial pilot |
| `openai_budget_sweep/hard_raw.jsonl` | 720 successful OpenAI receiver calls over ContextBench and ToolSandbox hard rows |

The corresponding summaries and sample manifests are in
`code/artifacts/openai_budget_sweep/`. The hard-sample run is summarized by:

```bash
python code/scripts/reproduce_paper.py openai-budget-sweep
```

## Sample sizes per benchmark (Qwen runs)

| Benchmark        | n_step per condition |
|------------------|----------------------|
| `bfcl`           | 100                  |
| `contextbench`   | 480                  |
| `repobench`      | 480                  |
| `swe_bench_lite` | 288                  |
| `tau_bench`      | 72 / 360             |
| `toolsandbox`    | 72 / 144 / 240 / 276 |

(Where multiple values are listed, the larger numbers correspond to the
main MA condition and SF condition; the smaller to the matched-sender
diagnostic and protocol-knob slices. See per-file row counts on disk.)

The manuscript-facing cells are reproduced by:

```bash
python code/scripts/reproduce_paper.py main-table-qwen --strict
python code/scripts/reproduce_paper.py sender-deltas --strict
python code/scripts/reproduce_paper.py openai-budget-sweep
```

Those commands apply the slice policies in `code/slice_manifests/paper_slices.json`
before computing means and Wilson intervals. Direct whole-file means remain
useful for inventory and exploratory checks, but they are not always identical
to manuscript cells when a file stores repeated diagnostic probes.

## Reproducing the paper's deltas and CIs

Each paper-level metric can be reproduced from these files alone:

- **Single-cell hit rates**: take the mean of `hit_joint` (or its components)
  over a `*.jsonl` file after applying the relevant manifest slice when one is
  specified for a manuscript cell.
- **Paired deltas (Δ_comm, Δ_cmp, Δ_strong)**: join two files on
  `(instance_id, step_t)`, compute `hit_joint` differences, then take the mean.
- **Paired bootstrap 95 % CIs**: resample the joined records with replacement
  5 000 times; report the 2.5 / 97.5 percentiles of the mean difference.
- **OpenAI model-family hit rates and Wilson CIs**: read derived summaries from
  `code/artifacts/openai_model_family_summary/`.
- **OpenAI hard-sample B-sweep**: read actual API-call rows from
  `data/openai_budget_sweep/hard_raw.jsonl` or the aggregate table from
  `code/artifacts/openai_budget_sweep/hard_summary.csv`.

A dependency-free implementation of Wilson intervals, paired deltas, and
bootstrap CIs is included under `code/scripts/release_harness/metrics.py`.

## Reuse

Released under the same MIT License as the rest of the repository.
