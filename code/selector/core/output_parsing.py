"""
core/output_parsing.py — Robust extraction of action JSON from model outputs.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)


def _balanced_object_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            cur = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif cur == "\\":
                    escape = True
                elif cur == '"':
                    in_string = False
                continue

            if cur == '"':
                in_string = True
            elif cur == "{":
                depth += 1
            elif cur == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : idx + 1])
                    break
    return candidates


def _try_parse_object(candidate: str) -> dict[str, Any] | None:
    candidate = candidate.strip()
    if not candidate:
        return None

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _score_object(parsed: dict[str, Any], preferred_keys: tuple[str, ...]) -> tuple[int, int, int]:
    keys = set(parsed.keys())
    preferred_hits = sum(1 for key in preferred_keys if key in keys)
    try:
        payload_size = len(json.dumps(parsed, ensure_ascii=False, default=str))
    except Exception:
        payload_size = len(repr(parsed))
    return (preferred_hits, len(keys), payload_size)


def _object_signature(parsed: dict[str, Any]) -> str:
    try:
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        return repr(parsed)


def _parsed_object_candidates(text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()

    direct = _try_parse_object(text)
    if direct is not None:
        serialized = _object_signature(direct)
        seen.add(serialized)
        candidates.append(direct)

    fenced = _FENCED_JSON_RE.findall(text)
    for candidate in reversed(fenced):
        parsed = _try_parse_object(candidate)
        if parsed is None:
            continue
        serialized = _object_signature(parsed)
        if serialized in seen:
            continue
        seen.add(serialized)
        candidates.append(parsed)

    balanced = _balanced_object_candidates(text)
    for candidate in balanced:
        parsed = _try_parse_object(candidate)
        if parsed is None:
            continue
        serialized = _object_signature(parsed)
        if serialized in seen:
            continue
        seen.add(serialized)
        candidates.append(parsed)

    return candidates


def extract_json_object(
    text: str,
    preferred_keys: tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """
    Extract the most likely JSON object from a model output.

    Handles:
    - raw JSON
    - fenced ```json blocks
    - prose or chain-of-thought before the object
    - Python-literal style dicts as a fallback
    """
    if not text:
        return None

    text = text.strip()
    candidates = _parsed_object_candidates(text)
    if not candidates:
        return None
    return max(candidates, key=lambda parsed: _score_object(parsed, preferred_keys))


def extract_action_prediction(text: str) -> dict[str, str] | None:
    """
    Extract and normalize an action prediction object.

    Normalizes:
    - action -> action_type
    - target -> target_artifact
    """
    parsed = extract_json_object(
        text,
        preferred_keys=("action_type", "action", "target_artifact", "target"),
    )
    if parsed is None:
        return None

    action_type = parsed.get("action_type", parsed.get("action", ""))
    target_artifact = parsed.get("target_artifact", parsed.get("target", ""))

    return {
        "action_type": str(action_type).strip(),
        "target_artifact": str(target_artifact).strip(),
    }


def snap_to_candidate_set(
    pred: dict[str, str],
    valid_actions: list[str],
    valid_artifacts: list[str],
) -> dict[str, str]:
    """
    Snap a prediction to the nearest valid candidate.

    Handles common hallucination patterns:
    - Suffix additions: "web_search_results" → "web_search"
    - Plural: "search_contacts" → "search_contact"  (if exact exists)
    - Prefix: "get_wifi_status_result" → "get_wifi_status"
    - Action hallucination: "web_search" → "run_test" (if not in valid list)

    Uses longest common prefix match, then substring containment, then
    Levenshtein-like edit distance as fallback.
    """
    import difflib

    action = pred.get("action_type", "")
    artifact = pred.get("target_artifact", "")

    # Snap action_type
    if action not in valid_actions:
        matches = difflib.get_close_matches(action, valid_actions, n=1, cutoff=0.5)
        action = matches[0] if matches else (valid_actions[0] if valid_actions else action)

    # Snap artifact — try exact first, then substring, then fuzzy
    if artifact not in valid_artifacts:
        # Try: does a valid artifact start with the prediction or vice versa?
        prefix_matches = [a for a in valid_artifacts if a.startswith(artifact) or artifact.startswith(a)]
        if prefix_matches:
            # Pick the one closest in length
            artifact = min(prefix_matches, key=lambda a: abs(len(a) - len(artifact)))
        else:
            matches = difflib.get_close_matches(artifact, valid_artifacts, n=1, cutoff=0.4)
            artifact = matches[0] if matches else artifact

    return {
        "action_type": action,
        "target_artifact": artifact,
    }
