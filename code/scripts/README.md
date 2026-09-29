# Reproduction Scripts

These scripts reproduce released paper numbers without rerunning any LLM.
They use only Python's standard library.

## Main commands

```bash
python code/scripts/reproduce_paper.py all --strict
python code/scripts/reproduce_paper.py main-table-qwen --strict
python code/scripts/reproduce_paper.py sender-deltas --strict
python code/scripts/reproduce_paper.py selector-ablation
python code/scripts/reproduce_paper.py selector-summary
python code/scripts/reproduce_paper.py openai-budget-sweep
```

## Utility commands

Score one JSONL file:

```bash
python code/scripts/score_records.py data/contextbench/qwen-ma-9B-from-80B.jsonl
```

Compute a paired delta on common `(instance_id, step_t)` keys:

```bash
python code/scripts/score_records.py \
  data/contextbench/qwen-ma-9B-from-80B.jsonl \
  data/contextbench/qwen-receiver-only.jsonl \
  --bootstrap 5000
```

Filter a file to the last 72 released rows:

```bash
python code/scripts/filter_records.py \
  --input data/toolsandbox/qwen-receiver-only.jsonl \
  --last-n 72 \
  --output out/toolsandbox_ro_last72.jsonl
```

Validate the paper slice manifest:

```bash
python code/scripts/build_slice_manifest.py --check
```

Run a small real OpenAI backend smoke across one released prompt shape per
benchmark and a chosen model list:

```bash
OPENAI_API_KEY=... python code/scripts/openai_benchmark_smoke.py \
  --models gpt-4o-mini gpt-5.4-mini gpt-5.4
```

This smoke test verifies API/backend compatibility only. It does not rerun the
full benchmark harness or replace the data-side reproduction commands above.

Summarize the released OpenAI receiver-side hard-sample message-budget sweep:

```bash
python code/scripts/reproduce_paper.py openai-budget-sweep
```

This command reads `code/artifacts/openai_budget_sweep/hard_summary.csv`, which
aggregates the recorded API calls in `data/openai_budget_sweep/hard_raw.jsonl`.
It does not call the API.

## Slice manifest

`code/slice_manifests/paper_slices.json` records the duplicate-handling and
slice policies needed to reproduce rounded manuscript values from the
released artifacts. Some diagnostic JSONL files intentionally include
repeated `(instance_id, step_t)` rows from repeated protocol probes, so
the manifest makes `first`, `last`, `seed`, and common-key choices explicit.
