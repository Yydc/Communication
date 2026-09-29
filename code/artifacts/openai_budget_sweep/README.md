# OpenAI Receiver-Side Message-Budget Sweep

This folder contains the derived artifacts for the live OpenAI receiver-side
budget sweep used for the appendix budget-fairness control. Raw API-call JSONL
records are stored under `../../../data/openai_budget_sweep/`.

The experiment reuses released Qwen sender messages and released receiver
prompt shapes, then varies only the visible coordination-message budget before
calling OpenAI receivers. It is a receiver-side robustness control, not a full
OpenAI sender rerun.

## Files

| File | Contents |
|---|---|
| `pilot_summary.csv` | Aggregated pilot hit rates by benchmark, receiver alias, and message budget. |
| `pilot_sample_manifest.csv` | The sampled released Qwen rows used by the pilot. |
| `hard_summary.csv` | Aggregated hard-sample hit rates by benchmark, receiver alias, and message budget. |
| `hard_sample_manifest.csv` | The sampled released Qwen rows used as the hard subset. The sample prioritizes released failures, sender-message truncation, and longer protocol inputs. |

Raw records:

| File | Contents |
|---|---|
| `../../../data/openai_budget_sweep/pilot_raw.jsonl` | Initial live pilot outputs for 120 successful calls. |
| `../../../data/openai_budget_sweep/hard_raw.jsonl` | Main appendix hard-sample outputs for 720 successful calls. |

## Actual Hard-Sweep Summary

The live run completed 720 API calls with no failed calls.

| Benchmark | Budget | gpt-5.4-nano | gpt-5.4-mini | gpt-5.4 | Mean |
|---|---:|---:|---:|---:|---:|
| ContextBench | 4 | 0.700 | 0.600 | 0.400 | 0.567 |
| ContextBench | 8 | 0.650 | 0.650 | 0.600 | 0.633 |
| ContextBench | 16 | 0.700 | 0.650 | 0.750 | 0.700 |
| ContextBench | 32 | 0.700 | 0.700 | 0.700 | 0.700 |
| ContextBench | 64 | 0.650 | 0.700 | 0.700 | 0.683 |
| ContextBench | 128 | 0.700 | 0.700 | 0.700 | 0.700 |
| ToolSandbox | 4 | 0.900 | 0.950 | 0.800 | 0.883 |
| ToolSandbox | 8 | 0.550 | 0.700 | 0.750 | 0.667 |
| ToolSandbox | 16 | 0.350 | 0.600 | 0.800 | 0.583 |
| ToolSandbox | 32 | 0.350 | 0.400 | 0.450 | 0.400 |
| ToolSandbox | 64 | 0.250 | 0.300 | 0.250 | 0.267 |
| ToolSandbox | 128 | 0.250 | 0.250 | 0.250 | 0.250 |

## Schema

`../../../data/openai_budget_sweep/hard_raw.jsonl` includes:

| Key | Meaning |
|---|---|
| `benchmark` | `ContextBench` or `ToolSandbox` |
| `receiver_model` | OpenAI receiver alias |
| `budget_tokens` | Visible sender-message budget used for this call |
| `instance_id`, `step_t` | Released source row identity |
| `gold_action`, `gold_artifact` | Scorer-side target fields |
| `action_type_pred`, `artifact_pred` | Parsed receiver prediction |
| `hit_action`, `hit_artifact`, `hit_joint` | Scoring outcomes |
| `input_tokens`, `output_tokens`, `latency_ms` | API-side accounting |
| `released_*` | Metadata from the released Qwen source row |
| `receiver_output` | Raw OpenAI receiver output |

## Reproduction

From the repository root:

```bash
python code/scripts/reproduce_paper.py openai-budget-sweep
```
