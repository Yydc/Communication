# Protocol selector

Reference implementation of the protocol selector (§3.2 / §4.6 / Figure 6
of the NeurIPS 2026 paper).

The runtime, paper-faithful entry point is **`selector.py`** —
`ProtocolSelector` implements Eq. (5) over the full
`{skip, compress, raw, escalate, refresh}` action set.

`selector_controller.py` is the threshold-based controller that generated the
released `selector-*` JSONL records under `../../data/`. New code should call
`selector.ProtocolSelector` rather than `selector_controller.SelectorController`
when reproducing the paper-level Eq. (5) selector.

## What the selector does

The paper-level control objective scores candidate protocol actions

```
α ∈ { skip, compress(B), raw, escalate(C'), refresh }
```

by an oracle-free deployed score

```
Ĵ_t(α)  =  Â_t(α)  −  τ̂_t(α)  −  η_c · c(α)  −  η_r · r̂_t(α)
```

and selects `argmax_α Ĵ_t(α)`. The tax offsets `τ̂` and the weights
`η_c, η_r` are fit once on a held-out calibration slice disjoint from the
reporting slice (`SelectorCalibration.fit`) and then frozen; `Â` is measured
by receiver probes at decision time, `c` from token counts, and `r̂` from
shared-state probes. The deployed score never observes gold artifacts or
actions at inference time.

`selector.py` provides one small class per surrogate so that ablations
and unit tests can swap one out without touching the rest of the
controller:

| Surrogate     | Class in `selector.py`             | What it estimates                                            |
|---------------|------------------------------------|--------------------------------------------------------------|
| `Â`           | `SurprisalGainEstimator`           | Surprisal-differential gain (or verifier-score proposal)     |
| `τ̂`          | `PairedTaxCalibrator`              | Paired RO/SF/MA tax read off the calibration table           |
| `c`           | `TokenCostModel`                   | Immediate prompt + completion token cost                     |
| `r̂`          | `SharedStateResidualEstimator`     | Finite-horizon residual risk from shared-state probes        |

One decision follows Appendix C.6 of the paper. `ProtocolSelector.step`
first generates and caches two real sender candidates: `compress`, which
summarizes the upstream context under the message budget, and `refresh`,
which summarizes the refreshed shared state. It then scores the five actions
with five no-message and four conditioned receiver probes, each capped at
four tokens; each probe pair uses the receiver that executes its action, and
the conditioned probe uses the action's actual input. `raw` appends the full
upstream context to the receiver observation, and `escalate` reuses the
compressed message with the stronger receiver. The selected action handler
(`SkipHandler`, `CompressHandler`, `RawHandler`, `EscalateHandler`,
`RefreshHandler`) then makes one final receiver generation, reusing the
cached candidate, and returns a `(ModelResponse, StepRecord)` pair whose
schema matches `../../data/*.jsonl`. The record's `total_*` token and latency
fields include both candidate generations, all probes, and the final call.

## Installation and requirements

```bash
cd code/selector
pip install -r requirements.txt        # all three providers + tiktoken
```

The gain surrogate `Â` is the clipped difference between the receiver's
conditioned and unconditioned leading-token log-probabilities, so the
receivers (including the stronger receiver used by `escalate`) must return
leading-token log-probs. `OpenAIBackend` returns them for non-reasoning models
such as `gpt-4o-mini`; `TogetherBackend` and `AnthropicBackend` do not, and
the estimator raises an error rather than substituting a value.

The released selector numbers are reproduced from stored records without
model calls (see "Reproducing selector numbers" below). For a real-API
compatibility check on released prompts, use
`../scripts/openai_benchmark_smoke.py`.

## File layout

```
selector/
├── README.md                 (this file)
├── requirements.txt          Optional per-provider dependencies
├── selector.py               ProtocolSelector — paper-faithful Ĵ_t selector with 5 actions
├── selector_controller.py        Threshold controller; generated the released selector-* data
├── agents/
│   ├── __init__.py
│   └── ma_sequential.py      Standard sender → receiver pipeline (the controller's inner agent)
├── core/
│   ├── __init__.py
│   ├── types.py              Request/response dataclasses, StepRecord
│   ├── tokenizer.py          BudgetEnforcer + EncodingFormatter
│   ├── output_parsing.py     JSON / action extraction utilities
│   ├── constraint_injection.py Candidate-set prompt constraints
│   ├── near_miss.py          Conservative parsed-output correction
│   └── model_interface.py    Abstract LLM backend (BaseModelBackend)
└── backends/
    ├── __init__.py
    ├── openai_backend.py     OpenAI chat-completions
    ├── together_backend.py   Together AI (OpenAI-compatible API)
    └── anthropic_backend.py  Anthropic Claude messages
```

The controller's source-code class name is `SelectorController`. The paper
refers to it generically as "the protocol selector" or "the deployed
controller"; the `agent_type` field in `../../data/*.jsonl` uses the prefix
`selector_*` to match the source.

## Calibration

`SelectorCalibration.fit` consumes a held-out calibration slice (disjoint
from the reporting slice) and returns the frozen `(η_c, η_r, τ̂_table)`
tuple used at inference time. The grid for `η_c, η_r` and the per-action
tax table are honest products of the calibration slice; no test labels
are read.

## Minimal usage

```python
from selector import ProtocolSelector, SelectorCalibration
from backends import OpenAIBackend, TogetherBackend, AnthropicBackend
from core.tokenizer import BudgetEnforcer

# Pick whichever backends fit your sender / receiver pair.
# All three constructors read their API key from an environment variable;
# none accept secrets as arguments.

# Example A: OpenAI sender + OpenAI receiver (paper §4.6 default)
sender   = OpenAIBackend("gpt-5.4")        # reads OPENAI_API_KEY
receiver = OpenAIBackend("gpt-4o-mini")    # reads OPENAI_API_KEY

# Example B: Together-hosted Qwen sender + Qwen receiver (paper Qwen rows)
# sender   = TogetherBackend("Qwen/Qwen3-Next-80B-A3B-Instruct")  # TOGETHER_API_KEY
# receiver = TogetherBackend("Qwen/Qwen3.5-9B")                    # TOGETHER_API_KEY

# Example C: Claude sender + Claude receiver (cross-family probe)
# sender   = AnthropicBackend("claude-sonnet-4-5-20250929")        # ANTHROPIC_API_KEY
# receiver = AnthropicBackend("claude-haiku-4-5-20251001")         # ANTHROPIC_API_KEY

# Static one-step setting of Appendix C.6 (eta_c = 1e-4 on raw token counts,
# eta_r = 0). SelectorCalibration.fit(records) fits these values and the tax
# table from held-out calibration records instead.
calib = SelectorCalibration(eta_c=1e-4, eta_r=0.0, tax_table={})

sel = ProtocolSelector(
    sender=sender,
    receiver=receiver,
    budget_enforcer=BudgetEnforcer(),
    budget=128,
    calibration=calib,
)

# One released ContextBench step: the full context seen in the SF protocol and
# the local view seen in the RO protocol, matched on (instance_id, step_t).
import json

def by_key(path):
    with open(path) as f:
        return {(r["instance_id"], r["step_t"]): r for r in map(json.loads, f)}

sf = by_key("../../data/contextbench/qwen-single-full.jsonl")
ro = by_key("../../data/contextbench/qwen-receiver-only.jsonl")
key = next(k for k in ro if k in sf)

response, step_record = sel.step(
    sender_context=sf[key]["receiver_input"],
    receiver_observation=ro[key]["receiver_input"],
    candidate_set=None,
    step_t=key[1],
)
```

The three backends are interchangeable wherever the controller expects a
`BaseModelBackend`. They each implement `generate(request: ModelRequest) ->
ModelResponse`; the controller does not call `score_candidates`. Only
`OpenAIBackend` returns the leading-token log-probs that `ProtocolSelector`
needs for its gain probes.

`step_record` is a `StepRecord` (defined in `core/types.py`) with the same
keys as the JSONL records under `data/`, including `hit_joint`, `hit_artifact`,
`hit_action`, `message_tokens`, etc.

## Backend notes

| Backend          | API spec                  | Key env var          | Reasoning models | Notes |
|------------------|---------------------------|----------------------|------------------|-------|
| OpenAIBackend    | OpenAI chat.completions   | `OPENAI_API_KEY`     | yes (`o3`, `o4-mini`, `gpt-5.4*`) | drops `temperature` and uses `max_completion_tokens` for reasoning models |
| TogetherBackend  | OpenAI-compatible (Together) | `TOGETHER_API_KEY` | n/a              | base URL `https://api.together.xyz/v1`; good for hosted Qwen / Llama / DeepSeek |
| AnthropicBackend | Anthropic messages        | `ANTHROPIC_API_KEY`  | uses `claude-*` thinking variants natively | system prompt is a top-level kwarg; tool-use blocks are not surfaced |

All three:
- read API keys only from environment variables (no secrets in args/code);
- raise a clear `RuntimeError` if the key env var is missing;
- raise a clear `ImportError` if the underlying SDK is not installed;
- return a uniform `ModelResponse` with `content`, `input_tokens`,
  `output_tokens`, `latency_ms`, `model_name`.

The per-cell summaries under `../artifacts/openai_model_family_summary/`
record the OpenAI model aliases used in the paper (for example `gpt-4o-mini`
and `gpt-5.4`). The strict-recoding records store the dated model identifier
returned by the API in `api_model` (for example `gpt-5.4-2026-03-05`). Together
AI and Anthropic backends are provided so re-running with open-weight or
Claude receivers is a one-line change.

## What is *not* included

- A full evaluation harness (benchmark loaders, parallel runners, batch
  budget guards). The records under `../../data/`, together with
  `../scripts/` and `../slice_manifests/`, reconstruct the reported released
  numeric cells without re-running models.
- A bounded-evaluation API client. The `BaseModelBackend` in `core/model_interface.py`
  defines the abstract interface the controller expects; a concrete OpenAI / vLLM /
  SGLang backend is also included for reference but is unused unless you
  re-run the experiments end-to-end.
- Calibration records for `SelectorCalibration.fit`.

## Selector records in this release

- Step records of the four-action threshold controller
  (`selector_controller.py`) are in `../../data/tau_bench/selector-*.jsonl`
  and `../../data/toolsandbox/selector-*.jsonl`.
- The static ContextBench selector ablation backing Figure 6(a) and Table 20
  is in `../artifacts/selector_ablation/`. It was computed offline from stored
  OpenAI RO/SF/MA records, not by running `ProtocolSelector`.

## Reproducing selector numbers from the released data

Without re-running any model:

```bash
cd ../..
python code/scripts/reproduce_paper.py selector-summary
python code/scripts/reproduce_paper.py selector-ablation
```

The commands print selector / static-baseline hit rates and the ContextBench
per-term selector ablation used in the paper.

## License

Released under the MIT License.
