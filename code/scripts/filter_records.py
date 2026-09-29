#!/usr/bin/env python3
"""Filter a released JSONL file using explicit paper slice selectors."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from release_harness.metrics import apply_selector, common_key_set, read_jsonl  # noqa: E402


def parse_selector(text: str) -> dict[str, Any]:
    if not text:
        return {}
    return json.loads(text)


def parse_where(values: list[str]) -> dict[str, Any]:
    where: dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"--where must be FIELD=VALUE, got {value!r}")
        field, raw = value.split("=", 1)
        try:
            where[field] = int(raw)
        except ValueError:
            where[field] = raw
    return where


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input JSONL path.")
    parser.add_argument("--output", help="Output JSONL path. Defaults to stdout.")
    parser.add_argument("--where", action="append", default=[], help="FIELD=VALUE filter.")
    parser.add_argument("--first-n", type=int)
    parser.add_argument("--last-n", type=int)
    parser.add_argument("--dedupe", choices=["first", "last"])
    parser.add_argument("--indices", help="Comma-separated zero-based row indices.")
    parser.add_argument(
        "--common-with",
        action="append",
        default=[],
        help="Other JSONL paths used to compute common instance_id/step_t keys.",
    )
    parser.add_argument(
        "--common-dedupe",
        choices=["first", "last"],
        help="Duplicate policy used before computing common keys.",
    )
    args = parser.parse_args()

    selector: dict[str, Any] = {}
    where = parse_where(args.where)
    if where:
        selector["where"] = where
    if args.first_n is not None:
        selector["first_n"] = args.first_n
    if args.last_n is not None:
        selector["last_n"] = args.last_n
    if args.dedupe:
        selector["dedupe"] = args.dedupe
    if args.indices:
        selector["indices"] = [int(part) for part in args.indices.split(",") if part]

    key_fields = ["instance_id", "step_t"]
    common_keys = None
    if args.common_with:
        common_keys = common_key_set(
            ROOT,
            [args.input] + args.common_with,
            key_fields=key_fields,
            dedupe=args.common_dedupe,
        )

    rows = apply_selector(
        read_jsonl(ROOT / args.input),
        selector,
        key_fields=key_fields,
        common_keys=common_keys,
    )

    out_lines = [json.dumps(row.record, ensure_ascii=False) for row in rows]
    payload = "\n".join(out_lines) + ("\n" if out_lines else "")
    if args.output:
        out_path = ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
    else:
        sys.stdout.write(payload)


if __name__ == "__main__":
    main()
