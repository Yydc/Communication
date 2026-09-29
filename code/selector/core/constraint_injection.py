"""Candidate-set constraint text shared by selector examples and scoring."""

from __future__ import annotations

import json

from core.types import CandidateSet


def build_candidate_constraint_block(candidate_set: CandidateSet) -> str:
    """Build the exact-choice constraint block used in released records."""

    artifacts = json.dumps(candidate_set.artifacts, ensure_ascii=False)
    actions = json.dumps(candidate_set.action_types, ensure_ascii=False)
    return (
        "\n\n[CRITICAL CONSTRAINT: You MUST select target_artifact EXACTLY "
        "from the list below. Do NOT add suffixes, plurals, or modify the "
        "name in any way. Copy the name character-for-character.\n"
        f"VALID target_artifact values: {artifacts}\n"
        f"VALID action_type values: {actions}]"
    )

