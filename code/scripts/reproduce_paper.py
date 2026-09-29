#!/usr/bin/env python3
"""Reproduce headline manuscript numbers from the released artifacts.

This script does not rerun LLMs. It rebuilds reported hit rates, deltas,
Wilson intervals, selector summaries, and selector/static-baseline summary
cells from JSONL/CSV records in data/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from release_harness.metrics import (  # noqa: E402
    apply_selector,
    common_key_set,
    hit_rate,
    read_csv,
    read_jsonl,
    wilson_interval,
)


MANIFEST_PATH = CODE_ROOT / "slice_manifests" / "paper_slices.json"


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open() as f:
        return json.load(f)


def resolve_common_keys(manifest: dict[str, Any], group_name: str) -> set[tuple[Any, ...]]:
    group = manifest["groups"][group_name]
    return common_key_set(
        ROOT,
        group["paths"],
        key_fields=manifest["default_key_fields"],
        dedupe=group.get("dedupe"),
        where=group.get("where"),
    )


def selected_rows(manifest: dict[str, Any], path: str, selector: dict[str, Any]):
    rows = read_jsonl(ROOT / path)
    common_keys = None
    if selector.get("common_group"):
        common_keys = resolve_common_keys(manifest, selector["common_group"])
    return apply_selector(
        rows,
        selector,
        key_fields=manifest["default_key_fields"],
        common_keys=common_keys,
    )


def fmt(x: float) -> str:
    return f"{x:.3f}"


def check_expected(name: str, actual: float, expected: float | None, strict: bool) -> bool:
    if expected is None:
        return True
    ok = abs(actual - expected) <= 5e-4
    status = "OK" if ok else "MISMATCH"
    print(f"    expected={fmt(expected)} actual={fmt(actual)} {status}")
    if strict and not ok:
        raise SystemExit(f"{name} mismatch: expected {expected}, got {actual}")
    return ok


def main_table_qwen(args) -> None:
    manifest = load_manifest()
    print("Qwen main-table cells from released slices")
    print("benchmark,protocol,n,hj,hits,wilson_lo,wilson_hi")
    for cell in manifest["main_table_qwen"]:
        rows = selected_rows(manifest, cell["path"], cell.get("selector", {}))
        hits = sum(bool(row.record.get("hit_joint", False)) for row in rows)
        hj = hits / len(rows) if rows else float("nan")
        lo, hi = wilson_interval(hits, len(rows))
        print(
            f"{cell['benchmark']},{cell['protocol']},{len(rows)},"
            f"{fmt(hj)},{hits},{fmt(lo)},{fmt(hi)}"
        )
        check_expected(
            f"{cell['benchmark']} {cell['protocol']}",
            hj,
            cell.get("expected_hj"),
            args.strict,
        )


def sender_deltas(args) -> None:
    manifest = load_manifest()
    print("Stronger-sender deltas Delta_strong = MA_main - MA_match")
    print("benchmark,n_main,n_match,main_hj,match_hj,delta")
    for item in manifest["sender_delta_claims"]:
        main_rows = selected_rows(manifest, item["main_path"], item.get("main_selector", {}))
        match_rows = selected_rows(manifest, item["match_path"], item.get("match_selector", {}))
        main_hj = hit_rate(main_rows)
        match_hj = hit_rate(match_rows)
        delta = main_hj - match_hj
        print(
            f"{item['benchmark']},{len(main_rows)},{len(match_rows)},"
            f"{fmt(main_hj)},{fmt(match_hj)},{delta:+.3f}"
        )
        check_expected(item["benchmark"], delta, item.get("expected_delta"), args.strict)


def openai_extension(_args) -> None:
    print("OpenAI model-family extension summary CSV cells")
    for path in sorted((CODE_ROOT / "artifacts" / "openai_model_family_summary").glob("*.csv")):
        rows = read_csv(path)
        print(f"\n{path.relative_to(ROOT)}")
        print("agent_type,n,hj,hj_wilson_lo,hj_wilson_hi,in_tok,out_tok")
        for row in rows:
            print(
                f"{row['agent_type']},{row['n']},{float(row['hj']):.3f},"
                f"{float(row['hj_wilson_lo']):.3f},"
                f"{float(row['hj_wilson_hi']):.3f},"
                f"{row['in_tok']},{row['out_tok']}"
            )


def openai_budget_sweep(_args) -> None:
    artifact_dir = CODE_ROOT / "artifacts" / "openai_budget_sweep"
    summary_path = artifact_dir / "hard_summary.csv"

    rows = read_csv(summary_path)
    print("OpenAI receiver-side hard-sample message-budget sweep")
    print("benchmark,receiver_model,budget_tokens,n_ok,n_error,hj,ha,hact,in_tok,out_tok")
    for row in rows:
        print(
            f"{row['benchmark']},{row['receiver_model']},{row['budget_tokens']},"
            f"{row['n_ok']},{row['n_error']},{float(row['hit_joint']):.3f},"
            f"{float(row['hit_artifact']):.3f},{float(row['hit_action']):.3f},"
            f"{row['input_tokens']},{row['output_tokens']}"
        )

    print("\nBudget means across receiver aliases")
    print("benchmark,budget_tokens,mean_hj")
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        key = (row["benchmark"], row["budget_tokens"])
        grouped.setdefault(key, []).append(float(row["hit_joint"]))
    for (benchmark, budget), values in sorted(
        grouped.items(), key=lambda item: (item[0][0], int(item[0][1]))
    ):
        print(f"{benchmark},{budget},{sum(values) / len(values):.3f}")


def selector_summary(_args) -> None:
    print("Selector / static-baseline records (selector_*)")
    print("benchmark,file,n,hj,hits,action_counts")
    for path in sorted((ROOT / "data").glob("*/selector-*.jsonl")):
        rows = read_jsonl(path)
        hits = sum(bool(row.record.get("hit_joint", False)) for row in rows)
        actions: dict[str, int] = {}
        for row in rows:
            action = row.record.get("repair_mode", "")
            actions[action] = actions.get(action, 0) + 1
        print(
            f"{path.parent.name},{path.name},{len(rows)},"
            f"{fmt(hits / len(rows))},{hits},{json.dumps(actions, sort_keys=True)}"
        )


def selector_ablation(_args) -> None:
    path = CODE_ROOT / "artifacts" / "selector_ablation" / "contextbench-gpt-4o-mini.csv"
    rows = read_csv(path)
    print("Selector ablation backing the Eq. (5) per-term study (Figure 6(a), Table 20)")
    print("variant,n,mean_hj,mean_tokens,n_skip,n_compress,n_raw")
    for row in rows:
        print(
            f"{row['variant']},{row['n']},{float(row['mean_hj']):.4f},"
            f"{float(row['mean_tokens']):.1f},{row['n_skip']},"
            f"{row['n_compress']},{row['n_raw']}"
        )


def inventory(_args) -> None:
    print("JSONL inventory")
    print("path,n,unique_instance_step,hj")
    key_fields = ["instance_id", "step_t"]
    for path in sorted((ROOT / "data").glob("*/*.jsonl")):
        rows = read_jsonl(path)
        keys = {
            tuple(row.record.get(field) for field in key_fields)
            for row in rows
        }
        print(
            f"{path.relative_to(ROOT)},{len(rows)},{len(keys)},"
            f"{fmt(hit_rate(rows))}"
        )


def all_sections(args) -> None:
    main_table_qwen(args)
    print()
    sender_deltas(args)
    print()
    selector_ablation(args)
    print()
    selector_summary(args)
    print()
    openai_extension(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "section",
        nargs="?",
        default="all",
        choices=[
            "all",
            "main-table-qwen",
            "sender-deltas",
            "openai-extension",
            "openai-budget-sweep",
            "selector-ablation",
            "selector-summary",
            "inventory",
        ],
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if manifest expected values do not match released data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dispatch = {
        "all": all_sections,
        "main-table-qwen": main_table_qwen,
        "sender-deltas": sender_deltas,
        "openai-extension": openai_extension,
        "openai-budget-sweep": openai_budget_sweep,
        "selector-ablation": selector_ablation,
        "selector-summary": selector_summary,
        "inventory": inventory,
    }
    dispatch[args.section](args)


if __name__ == "__main__":
    main()
