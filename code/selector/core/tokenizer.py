"""
core/tokenizer.py — Token counting & budget enforcement.

Key responsibility: ensure sender messages stay within B_t tokens.
Uses the model's tokenizer when `tokenizer_path` is given; otherwise it
estimates tokens as len(text) / 3.5. The released records were produced
with this estimate.
"""

from __future__ import annotations
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class BudgetEnforcer:
    """
    Counts tokens and truncates messages to stay within budget B.
    
    Usage:
        enforcer = BudgetEnforcer(tokenizer_path="/models/Qwen3.5-27B")
        msg, info = enforcer.enforce(raw_message, budget=512)
    """

    def __init__(self, tokenizer_path: str | None = None):
        self._tokenizer = None
        self._tokenizer_path = tokenizer_path

    def set_tokenizer_path(self, tokenizer_path: str | None):
        """Bind to a specific tokenizer path and reset cached tokenizer if it changed."""
        if tokenizer_path == self._tokenizer_path:
            return
        self._tokenizer_path = tokenizer_path
        self._tokenizer = None

    def _get_tokenizer(self):
        if self._tokenizer is None:
            if self._tokenizer_path:
                try:
                    from transformers import AutoTokenizer
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self._tokenizer_path, trust_remote_code=True
                    )
                    logger.info(f"Loaded tokenizer from {self._tokenizer_path}")
                except Exception as e:
                    logger.warning(f"Failed to load tokenizer: {e}. Using word-based estimate.")
                    self._tokenizer = "fallback"
            else:
                self._tokenizer = "fallback"
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        tok = self._get_tokenizer()
        if tok == "fallback":
            # Estimate: 1 token ≈ 3.5 characters.
            return max(1, int(len(text) / 3.5))
        return len(tok.encode(text))

    def enforce(self, raw_message: str, budget: int) -> tuple[str, dict]:
        """
        Truncate message to fit within budget.
        
        Returns:
            (truncated_message, info_dict)
            info_dict: {
                "original_tokens": int,
                "final_tokens": int,
                "truncated": bool,
            }
        """
        original_tokens = self.count_tokens(raw_message)

        if original_tokens <= budget:
            return raw_message, {
                "original_tokens": original_tokens,
                "final_tokens": original_tokens,
                "truncated": False,
            }

        # Truncate by decoding budget-many tokens
        tok = self._get_tokenizer()
        if tok == "fallback":
            # Word-level truncation as fallback
            words = raw_message.split()
            # Approximate: keep budget * 3.5 characters
            char_budget = int(budget * 3.5)
            truncated = raw_message[:char_budget]
        else:
            token_ids = tok.encode(raw_message)[:budget]
            truncated = tok.decode(token_ids, skip_special_tokens=True)

        final_tokens = self.count_tokens(truncated)
        return truncated, {
            "original_tokens": original_tokens,
            "final_tokens": final_tokens,
            "truncated": True,
        }


class EncodingFormatter:
    """
    Converts a structured coordination payload into different encoding formats.
    Used by Protocol P4 (encoding_rewrite).
    """

    @staticmethod
    def to_prose(payload: dict) -> str:
        """Free-form prose description."""
        import json

        def _stringify(value):
            if isinstance(value, (dict, list)):
                return json.dumps(value, ensure_ascii=False, sort_keys=True)
            return str(value)

        parts = []
        action = payload.get("action", payload.get("action_type"))
        target = payload.get("target", payload.get("target_artifact", payload.get("target_file")))
        if action:
            parts.append(f"The next action should be {_stringify(action)}.")
        if target:
            parts.append(f"The target is {_stringify(target)}.")
        if "rationale" in payload:
            parts.append(f"The reasoning is: {_stringify(payload['rationale'])}")
        if "context" in payload:
            parts.append(f"Additional context: {_stringify(payload['context'])}.")
        if "constraints" in payload:
            parts.append(f"Constraints: {_stringify(payload['constraints'])}.")
        return " ".join(parts)

    @staticmethod
    def to_bullet_schema(payload: dict) -> str:
        """Structured bullet-point schema."""
        import json

        lines = []
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    @staticmethod
    def to_canonical_json(payload: dict) -> str:
        """Canonical JSON with sorted keys."""
        import json
        return json.dumps(payload, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def to_decision_header_json(payload: dict) -> str:
        """Short decision header followed by canonical JSON payload."""
        action = payload.get("action_type", payload.get("action", ""))
        target = payload.get("target_artifact", payload.get("target", payload.get("target_file", "")))
        header = f"DECISION action_type={action} target_artifact={target}".strip()
        return f"{header}\n{EncodingFormatter.to_canonical_json(payload)}"

    @staticmethod
    def to_bullet_json_footer(payload: dict) -> str:
        """Short bullet summary with a machine-readable JSON footer."""
        action = payload.get("action_type", payload.get("action", ""))
        target = payload.get("target_artifact", payload.get("target", payload.get("target_file", "")))
        rationale = payload.get("rationale", "")
        constraints = payload.get("constraints", "")
        lines = [
            f"- action_type: {action}",
            f"- target_artifact: {target}",
        ]
        if rationale:
            lines.append(f"- rationale: {rationale}")
        if constraints:
            lines.append(f"- constraints: {constraints}")
        lines.append("JSON_FOOTER")
        lines.append(EncodingFormatter.to_canonical_json(payload))
        return "\n".join(lines)

    @staticmethod
    def format(payload: dict, encoding_format: str) -> str:
        """Dispatch to the right formatter."""
        if encoding_format == "prose":
            return EncodingFormatter.to_prose(payload)
        elif encoding_format == "bullet_schema":
            return EncodingFormatter.to_bullet_schema(payload)
        elif encoding_format == "canonical_json":
            return EncodingFormatter.to_canonical_json(payload)
        elif encoding_format == "decision_header_json":
            return EncodingFormatter.to_decision_header_json(payload)
        elif encoding_format == "bullet_json_footer":
            return EncodingFormatter.to_bullet_json_footer(payload)
        else:
            raise ValueError(f"Unknown encoding format: {encoding_format}")
