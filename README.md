# Receiver-Relative Bounded Coordination

Companion repository for the NeurIPS 2026 paper *Bayes-Sufficient Compression
Is Not Enough: How Does Communication Help Multi-Agent Systems?*

## Layout

The release is intentionally split into two main folders:

```
release/
├── code/             Reproduction scripts, selector code, manifests, and derived tables
├── data/             Raw model-run records only
├── README.md
└── LICENSE
```

- `data/` contains raw JSONL records from released model runs. It does not
  contain derived summaries, plots, or documentation.
- `code/` contains everything executable or derived: reproduction scripts,
  selector reference code, paper slice manifests, data cards, and aggregate
  CSV/JSON tables used for paper-facing summaries.

## Quick Start

Inspect raw records:

```python
import json
with open("data/contextbench/qwen-ma-9B-from-80B.jsonl") as f:
    records = [json.loads(line) for line in f]
print(len(records), "paired (instance, step) records")
print({k for r in records for k in r}.intersection({
    "hit_joint", "hit_artifact", "hit_action",
    "instance_id", "step_t", "message_tokens"
}))
```

Reproduce released numeric summaries without rerunning any model:

```bash
python code/scripts/reproduce_paper.py all --strict
python code/scripts/reproduce_paper.py main-table-qwen --strict
python code/scripts/reproduce_paper.py sender-deltas --strict
python code/scripts/reproduce_paper.py selector-ablation
python code/scripts/reproduce_paper.py selector-summary
python code/scripts/reproduce_paper.py openai-budget-sweep
```

`code/slice_manifests/paper_slices.json` records the duplicate-handling and
slice policies used by these commands. This is important for diagnostic files
that contain repeated `(instance_id, step_t)` rows from repeated protocol
probes.

## File Layout

```
release/
├── code/
│   ├── README.md
│   ├── artifacts/
│   │   ├── openai_budget_sweep/
│   │   ├── openai_model_family_summary/
│   │   └── selector_ablation/
│   ├── docs/
│   │   └── data_card.md
│   ├── scripts/
│   ├── selector/
│   └── slice_manifests/
└── data/
    ├── bfcl/
    ├── contextbench/
    ├── openai_budget_sweep/
    ├── repobench/
    ├── swe_bench_lite/
    ├── tau_bench/
    └── toolsandbox/
```

## Scope

This release excludes the general evaluation harness and plotting pipeline.
The included scripts reproduce the reported numeric cells from released raw
records and derived artifacts without re-running benchmark loaders or models.


## License

Released under the MIT License (see `LICENSE`).
