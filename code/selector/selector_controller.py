"""
selector_controller.py — threshold selector controller.

DATA LINEAGE NOTE
=================
This file contains the threshold-based controller that generated the released
`selector-*` JSONL records under `../../data/`. It is a finetuning-free
meta-controller over four actions (SEND / SKIP / REWRITE / ESCALATE), and is
included so the selector/static-baseline records have executable lineage.

For the runtime, paper-faithful realization of the deployed score

    Ĵ_t(α)  =  Â_t(α)  −  τ̂_t(α)  −  η_c · c(α)  −  η_r · r̂_t(α)

over the full {skip, compress, raw, escalate, refresh} action set
specified in Section 3.2, see `selector.py`.

A finetuning-free meta-controller that wraps a standard ma_sequential agent
and decides, at each step, whether to SEND / SKIP / REWRITE / ESCALATE
the sender message.  All decisions are threshold-based (no learned weights).

Diagnostic signals computed per step:
    1. communication_gain   — compare MA hit_rate vs RO baseline on this benchmark
    2. parse_reliability    — did the receiver's output parse into valid JSON?
    3. budget_pressure      — fraction of budget consumed by the raw message
    4. schema_fidelity      — does the sender message conform to canonical JSON schema?

Control actions:
    SEND      — forward the sender message to the receiver unchanged
    SKIP      — suppress the message; receiver acts on local observation only
    REWRITE   — re-encode the sender output as compact canonical JSON
    ESCALATE  — swap in a stronger receiver model (if one is registered)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from core.model_interface import BaseModelBackend
from core.output_parsing import extract_action_prediction, extract_json_object
from core.tokenizer import BudgetEnforcer, EncodingFormatter
from core.types import CandidateSet, ModelRequest, ModelResponse, StepRecord

logger = logging.getLogger(__name__)


# ── Configurable thresholds ──────────────────────────────────────────────

@dataclass
class SelectorThresholds:
    """All decision thresholds — override via agent_cfg["selector_thresholds"]."""

    # If the receiver-only baseline hit_rate on this benchmark exceeds this,
    # communication is likely noise: SKIP.
    ro_noise_threshold: float = 0.80

    # If the sender message fails JSON parse, REWRITE.
    # (no threshold — binary signal)

    # If message tokens / budget > this fraction, REWRITE with shorter format.
    budget_pressure_threshold: float = 0.95

    # If the estimated absorption gap (proxy: receiver parse failure rate over
    # recent steps) exceeds this, ESCALATE to stronger receiver.
    absorption_gap_threshold: float = 0.60

    # Minimum number of completed steps before the controller starts adapting.
    # Below this, always SEND (warm-up period).
    warmup_steps: int = 0


# ── Control actions ──────────────────────────────────────────────────────

class ControlAction:
    SEND = "SEND"
    SKIP = "SKIP"
    REWRITE = "REWRITE"
    ESCALATE = "ESCALATE"


# ── Threshold Selector Controller ───────────────────────────────────────────

CANONICAL_JSON_KEYS = ("action_type", "target_artifact", "rationale", "context", "constraints")

CANONICAL_JSON_SCHEMA = {
    "action_type", "target_artifact", "rationale", "context", "constraints",
}


def _message_parses_as_json(text: str) -> bool:
    """Return True if *text* contains a parseable JSON object."""
    obj = extract_json_object(text, preferred_keys=CANONICAL_JSON_KEYS)
    return obj is not None


def _message_has_schema_fidelity(text: str) -> bool:
    """
    Return True if message contains a JSON object with at least
    action_type and target_artifact keys.
    """
    obj = extract_json_object(text, preferred_keys=CANONICAL_JSON_KEYS)
    if obj is None:
        return False
    return "action_type" in obj and "target_artifact" in obj


def _compact_rewrite(text: str, budget_enforcer: BudgetEnforcer, budget: int) -> str:
    """
    Try to extract a canonical JSON payload from *text* and re-serialize
    it compactly.  Falls back to truncation if extraction fails.
    """
    obj = extract_json_object(text, preferred_keys=CANONICAL_JSON_KEYS)
    if obj is not None:
        # Keep only the canonical keys to stay compact
        compact = {k: obj[k] for k in CANONICAL_JSON_KEYS if k in obj}
        rewritten = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        # If still over budget, trim the longest value
        while budget_enforcer.count_tokens(rewritten) > budget and compact:
            longest_key = max(compact, key=lambda k: len(str(compact.get(k, ""))))
            val = str(compact[longest_key])
            compact[longest_key] = val[: len(val) // 2] + "..."
            rewritten = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        return rewritten

    # Fallback: just truncate
    truncated, _ = budget_enforcer.enforce(text, budget)
    return truncated


class SelectorController:
    """
    Threshold-based selector controller.

    Wraps an MASequentialAgent and intercepts its step() to apply
    per-step diagnostic → control-action logic.

    Compatible with the pipeline: exposes the same ``step()`` interface
    as MASequentialAgent so ``run_condition`` dispatches identically.
    """

    # NOTE: do NOT set requires_receiver_observation here.
    # run_condition uses hasattr(agent, "requires_receiver_observation") to detect
    # receiver-only agents.  selector is dispatched as MA (via sender_history).

    def __init__(
        self,
        sender: BaseModelBackend,
        receiver: BaseModelBackend,
        budget_enforcer: BudgetEnforcer,
        budget: int = 512,
        encoding_format: str = "canonical_json",
        rewrite_from_canonical_payload: bool = False,
        # selector-specific
        thresholds: SelectorThresholds | None = None,
        ro_hit_rate: float = 0.0,
        strong_receiver: BaseModelBackend | None = None,
        mode: str = "adaptive",
    ):
        from agents.ma_sequential import MASequentialAgent

        self.inner = MASequentialAgent(
            sender=sender,
            receiver=receiver,
            budget_enforcer=budget_enforcer,
            budget=budget,
            encoding_format=encoding_format,
            rewrite_from_canonical_payload=rewrite_from_canonical_payload,
        )
        self.budget_enforcer = budget_enforcer
        self.budget = budget
        self.thresholds = thresholds or SelectorThresholds()
        self.ro_hit_rate = ro_hit_rate
        self.strong_receiver = strong_receiver
        self.mode = mode.removeprefix("selector_")
        # Internal values: "adaptive" | "always_send" | "always_skip" | "always_rewrite".

        # Running diagnostics
        self._step_count = 0
        self._parse_failures = 0

        # For duck-typing in run_condition (sender_history check)
        self.sender_history: list[dict] = []

    def reset(self):
        self.inner.reset()
        self.sender_history = []
        self._step_count = 0
        self._parse_failures = 0

    def set_budget(self, budget: int):
        self.budget = budget
        self.inner.set_budget(budget)

    def set_encoding_format(self, fmt: str):
        self.inner.set_encoding_format(fmt)

    def swap_receiver(self, new_receiver: BaseModelBackend):
        self.inner.swap_receiver(new_receiver)

    # ── Diagnostic signals ───────────────────────────────────────────

    def _diagnose(self, raw_message: str, receiver_output: str) -> dict:
        """Compute all diagnostic signals for the current step."""
        msg_tokens = self.budget_enforcer.count_tokens(raw_message)
        budget_pressure = msg_tokens / max(self.budget, 1)

        schema_ok = _message_has_schema_fidelity(raw_message)
        parse_ok = _message_parses_as_json(raw_message)
        receiver_parsed = extract_action_prediction(receiver_output) is not None

        # Running parse failure rate (proxy for absorption gap)
        total = self._step_count + 1
        fail_rate = self._parse_failures / max(total, 1)

        return {
            "communication_gain_positive": self.ro_hit_rate < self.thresholds.ro_noise_threshold,
            "ro_hit_rate": self.ro_hit_rate,
            "parse_ok": parse_ok,
            "schema_ok": schema_ok,
            "receiver_parsed": receiver_parsed,
            "budget_pressure": budget_pressure,
            "absorption_gap_proxy": fail_rate,
        }

    def _select_action(self, diag: dict) -> str:
        """
        Threshold-based control action selection.

        Priority order:
          1. Static overrides (always_send, always_skip, always_rewrite)
          2. Noise regime → SKIP
          3. Parse failure → REWRITE
          4. Budget violation → REWRITE
          5. Absorption gap → ESCALATE
          6. Default → SEND
        """
        if self.mode == "always_send":
            return ControlAction.SEND
        if self.mode == "always_skip":
            return ControlAction.SKIP
        if self.mode == "always_rewrite":
            return ControlAction.REWRITE

        # Warm-up: always SEND for the first N steps
        if self._step_count < self.thresholds.warmup_steps:
            return ControlAction.SEND

        # 1. Noise regime: RO baseline is already very good
        if not diag["communication_gain_positive"]:
            return ControlAction.SKIP

        # 2. Message failed to parse as JSON → REWRITE
        if not diag["parse_ok"]:
            return ControlAction.REWRITE

        # 3. Budget pressure too high → REWRITE with compact format
        if diag["budget_pressure"] > self.thresholds.budget_pressure_threshold:
            return ControlAction.REWRITE

        # 4. High absorption gap → ESCALATE (if strong receiver available)
        if (
            diag["absorption_gap_proxy"] > self.thresholds.absorption_gap_threshold
            and self.strong_receiver is not None
        ):
            return ControlAction.ESCALATE

        # 5. Default: send message as-is
        return ControlAction.SEND

    # ── Main step ────────────────────────────────────────────────────

    def step(
        self,
        sender_context: str,
        receiver_observation: str,
        candidate_set: Optional[CandidateSet] = None,
        step_t: int = 0,
    ) -> tuple[ModelResponse, StepRecord]:
        """
        Execute one selector-controlled coordination step.

        For SKIP: runs receiver-only (no sender call, saves API cost).
        For REWRITE: runs sender, rewrites message, then receiver.
        For ESCALATE: runs sender, then strong receiver.
        For SEND: delegates to inner ma_sequential.step() unchanged.
        """
        # ── Pre-decision: we need the sender message to diagnose ──
        # For SKIP mode, we can short-circuit without calling the sender.
        if self.mode == "always_skip":
            return self._run_skip(receiver_observation, candidate_set, step_t)

        # For adaptive mode, we may SKIP if noise regime is detected.
        # Check noise regime BEFORE calling sender to save API cost.
        if self.mode == "adaptive" and not self._is_warmup():
            if self.ro_hit_rate >= self.thresholds.ro_noise_threshold:
                return self._run_skip(receiver_observation, candidate_set, step_t)

        # ── Run the inner ma_sequential step ──
        response, record = self.inner.step(
            sender_context, receiver_observation, candidate_set, step_t,
        )

        # ── Post-step diagnostics ──
        raw_message = record.message_raw or ""
        receiver_output = record.receiver_output or ""
        diag = self._diagnose(raw_message, receiver_output)

        # Update running stats
        self._step_count += 1
        if not diag["receiver_parsed"]:
            self._parse_failures += 1

        # Keep sender_history in sync for duck-typing
        self.sender_history = list(self.inner.sender_history)

        action = self._select_action(diag)

        # ── Apply control action retroactively ──
        # For SEND: nothing to do, the step already ran normally.
        if action == ControlAction.SEND:
            record.agent_type = self._agent_type()
            record.repair_mode = f"selector_action=SEND"
            return response, record

        # For REWRITE: we already have the sender output. Rewrite and re-run receiver.
        if action == ControlAction.REWRITE:
            return self._apply_rewrite(
                raw_message, receiver_observation, candidate_set, step_t, record,
            )

        # For ESCALATE: re-run receiver with strong model
        if action == ControlAction.ESCALATE and self.strong_receiver is not None:
            return self._apply_escalate(
                raw_message, receiver_observation, candidate_set, step_t, record,
            )

        # Fallback: return as SEND
        record.agent_type = self._agent_type()
        record.repair_mode = f"selector_action={action}"
        return response, record

    def _is_warmup(self) -> bool:
        return self._step_count < self.thresholds.warmup_steps

    def _agent_type(self) -> str:
        return f"selector_{self.mode}"

    def _run_skip(
        self,
        receiver_observation: str,
        candidate_set: Optional[CandidateSet],
        step_t: int,
    ) -> tuple[ModelResponse, StepRecord]:
        """Run receiver-only (no sender message)."""
        from agents.ma_sequential import RECEIVER_SYSTEM_PROMPT

        messages = [{"role": "system", "content": RECEIVER_SYSTEM_PROMPT}]

        constraint_block = ""
        if candidate_set is not None:
            from core.constraint_injection import build_candidate_constraint_block
            constraint_block = build_candidate_constraint_block(candidate_set)

        receiver_input = (
            f"[Your local observation]\n{receiver_observation}\n\n"
            f"[No coordination message available. Use only your local observation.]"
            f"{constraint_block}"
        )
        messages.append({"role": "user", "content": receiver_input})

        request = ModelRequest(
            messages=messages,
            model_name=self.inner.receiver.model_name,
            max_tokens=4096,
            temperature=0.0,
            logprobs=candidate_set is not None,
        )
        response = self.inner.receiver.generate(request)

        record = StepRecord(
            agent_type=self._agent_type(),
            receiver_model=self.inner.receiver.model_name,
            sender_model=self.inner.sender.model_name,
            budget_tokens=self.budget,
            encoding_format=self.inner.encoding_format,
            step_t=step_t,
            candidate_set_size=candidate_set.total_labels if candidate_set else 0,
            message_raw="",
            message_tokens=0,
            receiver_input=receiver_input,
            receiver_input_tokens=response.input_tokens,
            receiver_output=response.content,
            receiver_output_tokens=response.output_tokens,
            total_input_tokens=response.input_tokens,
            total_output_tokens=response.output_tokens,
            latency_ms=response.latency_ms,
            repair_mode="selector_action=SKIP",
        )

        # Score coordination target
        if candidate_set is not None:
            record = self.inner._score_coordination(
                messages, response, receiver_observation, candidate_set, record,
            )

        self._step_count += 1
        return response, record

    def _apply_rewrite(
        self,
        raw_message: str,
        receiver_observation: str,
        candidate_set: Optional[CandidateSet],
        step_t: int,
        original_record: StepRecord,
    ) -> tuple[ModelResponse, StepRecord]:
        """Rewrite the sender message as compact canonical JSON, then re-run receiver."""
        from agents.ma_sequential import RECEIVER_SYSTEM_PROMPT

        rewritten = _compact_rewrite(raw_message, self.budget_enforcer, self.budget)

        messages = [{"role": "system", "content": RECEIVER_SYSTEM_PROMPT}]

        constraint_block = ""
        if candidate_set is not None:
            from core.constraint_injection import build_candidate_constraint_block
            constraint_block = build_candidate_constraint_block(candidate_set)

        receiver_input = (
            f"[Your local observation]\n{receiver_observation}\n\n"
            f"[Coordination message from sender]\n{rewritten}"
            f"{constraint_block}"
        )
        messages.append({"role": "user", "content": receiver_input})

        request = ModelRequest(
            messages=messages,
            model_name=self.inner.receiver.model_name,
            max_tokens=4096,
            temperature=0.0,
            logprobs=candidate_set is not None,
        )
        response = self.inner.receiver.generate(request)

        # Build a new record that captures both the original sender call
        # and the rewrite + re-run of the receiver
        record = StepRecord(
            agent_type=self._agent_type(),
            sender_model=self.inner.sender.model_name,
            receiver_model=self.inner.receiver.model_name,
            budget_tokens=self.budget,
            encoding_format="canonical_json",  # rewritten format
            step_t=step_t,
            candidate_set_size=candidate_set.total_labels if candidate_set else 0,
            message_raw=rewritten,
            message_tokens=self.budget_enforcer.count_tokens(rewritten),
            receiver_input=receiver_input,
            receiver_input_tokens=response.input_tokens,
            receiver_output=response.content,
            receiver_output_tokens=response.output_tokens,
            # Total cost includes original sender call + new receiver call
            total_input_tokens=(
                original_record.total_input_tokens
                - original_record.receiver_input_tokens
                + response.input_tokens
            ),
            total_output_tokens=(
                original_record.total_output_tokens
                - original_record.receiver_output_tokens
                + response.output_tokens
            ),
            latency_ms=original_record.latency_ms + response.latency_ms,
            repair_mode="selector_action=REWRITE",
        )

        if candidate_set is not None:
            record = self.inner._score_coordination(
                messages, response, receiver_observation, candidate_set, record,
            )

        return response, record

    def _apply_escalate(
        self,
        raw_message: str,
        receiver_observation: str,
        candidate_set: Optional[CandidateSet],
        step_t: int,
        original_record: StepRecord,
    ) -> tuple[ModelResponse, StepRecord]:
        """Re-run with the strong receiver model."""
        from agents.ma_sequential import RECEIVER_SYSTEM_PROMPT

        if self.strong_receiver is None:
            # No strong receiver registered — fall back to SEND
            original_record.repair_mode = "selector_action=ESCALATE_FALLBACK_SEND"
            original_record.agent_type = self._agent_type()
            return ModelResponse(
                content=original_record.receiver_output,
                input_tokens=original_record.receiver_input_tokens,
                output_tokens=original_record.receiver_output_tokens,
                latency_ms=original_record.latency_ms,
            ), original_record

        messages = [{"role": "system", "content": RECEIVER_SYSTEM_PROMPT}]

        constraint_block = ""
        if candidate_set is not None:
            from core.constraint_injection import build_candidate_constraint_block
            constraint_block = build_candidate_constraint_block(candidate_set)

        receiver_input = (
            f"[Your local observation]\n{receiver_observation}\n\n"
            f"[Coordination message from sender]\n{raw_message}"
            f"{constraint_block}"
        )
        messages.append({"role": "user", "content": receiver_input})

        request = ModelRequest(
            messages=messages,
            model_name=self.strong_receiver.model_name,
            max_tokens=4096,
            temperature=0.0,
            logprobs=candidate_set is not None,
        )
        response = self.strong_receiver.generate(request)

        record = StepRecord(
            agent_type=self._agent_type(),
            sender_model=self.inner.sender.model_name,
            receiver_model=self.strong_receiver.model_name,
            budget_tokens=self.budget,
            encoding_format=self.inner.encoding_format,
            step_t=step_t,
            candidate_set_size=candidate_set.total_labels if candidate_set else 0,
            message_raw=raw_message,
            message_tokens=self.budget_enforcer.count_tokens(raw_message),
            receiver_input=receiver_input,
            receiver_input_tokens=response.input_tokens,
            receiver_output=response.content,
            receiver_output_tokens=response.output_tokens,
            total_input_tokens=(
                original_record.total_input_tokens
                - original_record.receiver_input_tokens
                + response.input_tokens
            ),
            total_output_tokens=(
                original_record.total_output_tokens
                - original_record.receiver_output_tokens
                + response.output_tokens
            ),
            latency_ms=original_record.latency_ms + response.latency_ms,
            escalation_applied=True,
            repair_mode="selector_action=ESCALATE",
        )

        if candidate_set is not None:
            record = self.inner._score_coordination(
                messages, response, receiver_observation, candidate_set, record,
                receiver_backend=self.strong_receiver,
            )

        return response, record
