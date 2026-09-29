"""Shared metric and slicing helpers for the code/data release.

The helpers are intentionally dependency-free so users can run the
reproduction scripts with a stock Python 3 installation.
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class IndexedRecord:
    """One JSONL row plus its stable zero-based row index."""

    index: int
    record: dict[str, Any]


def read_jsonl(path: Path) -> list[IndexedRecord]:
    rows: list[IndexedRecord] = []
    with path.open() as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            rows.append(IndexedRecord(index=index, record=json.loads(line)))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def hit_rate(rows: Iterable[IndexedRecord], metric: str = "hit_joint") -> float:
    selected = list(rows)
    if not selected:
        return float("nan")
    return sum(bool(row.record.get(metric, False)) for row in selected) / len(selected)


def mean_numeric(rows: Iterable[IndexedRecord], field: str) -> float:
    selected = list(rows)
    if not selected:
        return float("nan")
    return sum(float(row.record.get(field, 0.0) or 0.0) for row in selected) / len(selected)


def key_for(row: IndexedRecord, key_fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.record.get(field) for field in key_fields)


def dedupe_rows(
    rows: list[IndexedRecord],
    key_fields: list[str],
    keep: str,
) -> list[IndexedRecord]:
    if keep not in {"first", "last"}:
        raise ValueError(f"Unknown duplicate policy: {keep!r}")
    by_key: dict[tuple[Any, ...], IndexedRecord] = {}
    if keep == "first":
        for row in rows:
            by_key.setdefault(key_for(row, key_fields), row)
    else:
        for row in rows:
            by_key[key_for(row, key_fields)] = row
    return list(by_key.values())


def apply_selector(
    rows: list[IndexedRecord],
    selector: dict[str, Any],
    *,
    key_fields: list[str],
    common_keys: set[tuple[Any, ...]] | None = None,
) -> list[IndexedRecord]:
    """Apply a small manifest selector to one JSONL file."""

    selected = list(rows)

    for field, value in selector.get("where", {}).items():
        selected = [row for row in selected if row.record.get(field) == value]

    if selector.get("dedupe"):
        selected = dedupe_rows(selected, key_fields, selector["dedupe"])

    if common_keys is not None:
        selected = [row for row in selected if key_for(row, key_fields) in common_keys]

    if "first_n" in selector:
        selected = selected[: int(selector["first_n"])]
    if "last_n" in selector:
        selected = selected[-int(selector["last_n"]) :]

    if "indices" in selector:
        allowed = {int(i) for i in selector["indices"]}
        selected = [row for row in selected if row.index in allowed]

    return selected


def common_key_set(
    root: Path,
    paths: list[str],
    *,
    key_fields: list[str],
    dedupe: str | None = None,
    where: dict[str, Any] | None = None,
) -> set[tuple[Any, ...]]:
    common: set[tuple[Any, ...]] | None = None
    for rel_path in paths:
        rows = read_jsonl(root / rel_path)
        if where:
            for field, value in where.items():
                rows = [row for row in rows if row.record.get(field) == value]
        if dedupe:
            rows = dedupe_rows(rows, key_fields, dedupe)
        keys = {key_for(row, key_fields) for row in rows}
        common = keys if common is None else common & keys
    return common or set()


def wilson_interval(hits: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = hits / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt((phat * (1.0 - phat) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def paired_delta(
    left: list[IndexedRecord],
    right: list[IndexedRecord],
    *,
    key_fields: list[str],
    metric: str = "hit_joint",
) -> tuple[float, int, list[float]]:
    left_by_key = {key_for(row, key_fields): row for row in left}
    right_by_key = {key_for(row, key_fields): row for row in right}
    keys = sorted(set(left_by_key) & set(right_by_key))
    diffs = [
        float(bool(left_by_key[key].record.get(metric, False)))
        - float(bool(right_by_key[key].record.get(metric, False)))
        for key in keys
    ]
    if not diffs:
        return float("nan"), 0, []
    return sum(diffs) / len(diffs), len(diffs), diffs


def bootstrap_ci(
    diffs: list[float],
    *,
    n_boot: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if not diffs:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(diffs)
    means: list[float] = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * (n_boot - 1))]
    hi = means[int((1 - alpha / 2) * (n_boot - 1))]
    return lo, hi

