"""Near-miss correction for parsed action/artifact strings."""

from __future__ import annotations

import difflib


def apply_near_miss_correction(value: str, valid_values: list[str]) -> tuple[str, str]:
    """Return the closest valid value and a short correction reason.

    The correction is conservative: exact matches are preserved; otherwise
    we try prefix/substring matches before falling back to difflib.
    """

    if not value or not valid_values:
        return value, "empty"
    if value in valid_values:
        return value, "exact"

    prefix_matches = [
        candidate
        for candidate in valid_values
        if candidate.startswith(value) or value.startswith(candidate)
    ]
    if prefix_matches:
        best = min(prefix_matches, key=lambda item: abs(len(item) - len(value)))
        return best, "prefix"

    contains_matches = [
        candidate
        for candidate in valid_values
        if candidate in value or value in candidate
    ]
    if contains_matches:
        best = min(contains_matches, key=lambda item: abs(len(item) - len(value)))
        return best, "contains"

    matches = difflib.get_close_matches(value, valid_values, n=1, cutoff=0.4)
    if matches:
        return matches[0], "fuzzy"
    return value, "unchanged"

