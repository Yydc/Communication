#!/usr/bin/env python3
"""Run a small OpenAI backend smoke across released benchmark prompt shapes.

This is not a benchmark rerun. It takes one released record per benchmark,
uses the record's receiver input as a compact task context, and runs one
selector step with each requested OpenAI model. The goal is to verify that
the release's real OpenAI backend path works across benchmark-specific
prompt shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
SELECTOR_DIR = CODE_ROOT / "selector"
sys.path.insert(0, str(SELECTOR_DIR))

from backends import OpenAIBackend  # noqa: E402
from core.tokenizer import BudgetEnforcer  # noqa: E402
from selector_controller import SelectorController, SelectorThresholds  # noqa: E402


DEFAULT_MODELS = ["gpt-4o-mini", "gpt-5.4-mini", "gpt-5.4"]

BENCHMARK_SOURCES = {
    "bfcl": "data/bfcl/qwen-receiver-only.jsonl",
    "contextbench": "data/contextbench/qwen-receiver-only.jsonl",
    "repobench": "data/repobench/qwen-receiver-only.jsonl",
    "swe_bench_lite": "data/swe_bench_lite/qwen-receiver-only.jsonl",
    "tau_bench": "data/tau_bench/qwen-receiver-only.jsonl",
    "toolsandbox": "data/toolsandbox/qwen-receiver-only.jsonl",
}


def load_first_useful_record(path: Path) -> dict:
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("receiver_input") or record.get("receiver_output"):
                return record
    raise RuntimeError(f"No useful records found in {path}")


def compact_observation(record: dict, max_chars: int) -> str:
    text = record.get("receiver_input") or record.get("receiver_output") or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated for smoke test]..."


def run_case(benchmark: str, record: dict, model: str, max_chars: int) -> dict:
    backend = OpenAIBackend(model)
    ctrl = SelectorController(
        sender=backend,
        receiver=backend,
        budget_enforcer=BudgetEnforcer(),
        budget=48,
        encoding_format="canonical_json",
        thresholds=SelectorThresholds(),
        mode="adaptive",
    )

    receiver_observation = compact_observation(record, max_chars)
    sender_context = (
        f"Benchmark: {benchmark}\n"
        "You are given a released receiver observation from an agentic "
        "next-action task. Produce a compact coordination message for the "
        "receiver in JSON with action_type and target_artifact.\n\n"
        f"{receiver_observation}"
    )

    response, step_record = ctrl.step(
        sender_context=sender_context,
        receiver_observation=receiver_observation,
        candidate_set=None,
        step_t=int(record.get("step_t", 0) or 0),
    )

    return {
        "benchmark": benchmark,
        "model": model,
        "action": step_record.repair_mode,
        "message_tokens": step_record.message_tokens,
        "receiver_input_tokens": step_record.receiver_input_tokens,
        "receiver_output_tokens": step_record.receiver_output_tokens,
        "receiver_output_preview": (response.content or "").replace("\n", " ")[:100],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=list(BENCHMARK_SOURCES),
        choices=list(BENCHMARK_SOURCES),
    )
    args = parser.parse_args()

    failures: list[tuple[str, str, str]] = []
    print("benchmark,model,status,action,msg_tok,in_tok,out_tok,preview")
    for benchmark in args.benchmarks:
        record = load_first_useful_record(ROOT / BENCHMARK_SOURCES[benchmark])
        for model in args.models:
            try:
                result = run_case(benchmark, record, model, args.max_chars)
                print(
                    f"{benchmark},{model},ok,{result['action']},"
                    f"{result['message_tokens']},"
                    f"{result['receiver_input_tokens']},"
                    f"{result['receiver_output_tokens']},"
                    f"{json.dumps(result['receiver_output_preview'])}"
                )
            except Exception as exc:
                failures.append((benchmark, model, type(exc).__name__))
                print(
                    f"{benchmark},{model},error,,,,,"
                    f"{json.dumps(type(exc).__name__ + ': ' + str(exc))}"
                )
    if failures:
        print(
            f"error: {len(failures)} OpenAI smoke case(s) failed",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
