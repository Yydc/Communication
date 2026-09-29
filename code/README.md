# Code and Derived Artifacts

This folder contains all non-raw release material:

| Path | Contents |
|---|---|
| `scripts/` | Reproduction, scoring, slicing, and smoke-test scripts |
| `selector/` | Minimal selector reference implementation |
| `slice_manifests/` | Paper-facing slice and duplicate-handling policies |
| `artifacts/` | CSV/JSON summaries aggregated from recorded runs |
| `docs/data_card.md` | Data schema and raw-record documentation |

The raw model-run records are kept under `../data/`.

Main offline reproduction commands from the repository root:

```bash
python code/scripts/reproduce_paper.py all --strict
python code/scripts/reproduce_paper.py main-table-qwen --strict
python code/scripts/reproduce_paper.py sender-deltas --strict
python code/scripts/reproduce_paper.py selector-ablation
python code/scripts/reproduce_paper.py selector-summary
python code/scripts/reproduce_paper.py openai-budget-sweep
```

