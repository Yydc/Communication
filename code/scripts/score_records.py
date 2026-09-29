#!/usr/bin/env python3
"""Score released JSONL records and paired deltas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = ROOT / "code"
sys.path.insert(0, str(CODE_ROOT / "scripts"))

from release_harness.metrics import (  # noqa: E402
    bootstrap_ci,
    hit_rate,
    paired_delta,
    read_jsonl,
    wilson_interval,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="One path for hit rate, two for paired delta.")
    parser.add_argument("--metric", default="hit_joint")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if len(args.paths) == 1:
        rows = read_jsonl(ROOT / args.paths[0])
        hits = sum(bool(row.record.get(args.metric, False)) for row in rows)
        lo, hi = wilson_interval(hits, len(rows))
        print(f"path={args.paths[0]}")
        print(f"n={len(rows)}")
        print(f"{args.metric}={hit_rate(rows, args.metric):.6f}")
        print(f"hits={hits}")
        print(f"wilson95=[{lo:.6f}, {hi:.6f}]")
        return

    if len(args.paths) == 2:
        left = read_jsonl(ROOT / args.paths[0])
        right = read_jsonl(ROOT / args.paths[1])
        delta, n, diffs = paired_delta(
            left,
            right,
            key_fields=["instance_id", "step_t"],
            metric=args.metric,
        )
        print(f"left={args.paths[0]}")
        print(f"right={args.paths[1]}")
        print(f"paired_n={n}")
        print(f"delta={delta:.6f}")
        if args.bootstrap:
            lo, hi = bootstrap_ci(diffs, n_boot=args.bootstrap, seed=args.seed)
            print(f"bootstrap95=[{lo:.6f}, {hi:.6f}]")
        return

    raise SystemExit("Pass either one JSONL path or two JSONL paths.")


if __name__ == "__main__":
    main()
