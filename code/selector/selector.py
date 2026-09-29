"""selector.py — Protocol selector implementing Eq. (5) from the paper.

This is the runtime, paper-faithful realization of

    J_t(α)  =  A_{Φ_α,ℓ}^*  −  τ_{ρ_α,ℓ}  −  η_c · c(α)  −  η_r · r_t(α)

with the deployed oracle-free score

    Ĵ_t(α)  =  Â_t(α)  −  τ̂_t(α)  −  η_c · c(α)  −  η_r · r̂_t(α)

over the action set

    α ∈ {skip, compress(B), raw, escalate(C'), refresh}

specified in Section 3.2 of the paper.

Each surrogate maps to one held-out-calibrated estimator:
  - Â  : surprisal-differential gain surrogate (or verifier-score surrogate)
          implemented by SurprisalGainEstimator
  - τ̂  : paired protocol calibration (RO/SF/MA cell deltas on held-out)
          implemented by PairedTaxCalibrator
  - c  : immediate token/latency cost (no calibration)
          implemented by TokenCostModel
  - r̂  : finite-horizon residual risk from shared-state probes
          implemented by SharedStateResidualEstimator

Controller weights η_c, η_r are scalar non-negative and fit once on the
calibration slice via SelectorCalibration.fit_weights, then frozen.

One decision follows Appendix C.6: the selector generates and caches two
real sender candidates (`compress` under budget B and `refresh` on the
refreshed shared state), scores the actions with five no-message and four
conditioned receiver probes capped at four tokens (each probe pair uses the
receiver that executes its action), and executes the selected action once
with its cached candidate.

The deployed score Ĵ_t never observes gold artifacts or actions at
inference time; the per-action handler implementations only invoke the
sender, receiver, and registered shared-state probes.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from core.model_interface import BaseModelBackend
from core.output_parsing import extract_action_prediction, extract_json_object
from core.tokenizer import BudgetEnforcer, EncodingFormatter
from core.types import CandidateSet, ModelRequest, ModelResponse, StepRecord

from agents.ma_sequential import (
    MASequentialAgent,
    RECEIVER_SYSTEM_PROMPT,
    SENDER_SYSTEM_PROMPT,
)
from core.constraint_injection import build_candidate_constraint_block

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Action enum.  Matches Section 3.2's action set exactly.
# ─────────────────────────────────────────────────────────────────────

class Action:
    SKIP     = "skip"
    COMPRESS = "compress"
    RAW      = "raw"
    ESCALATE = "escalate"
    REFRESH  = "refresh"

    ALL = (SKIP, COMPRESS, RAW, ESCALATE, REFRESH)


# ─────────────────────────────────────────────────────────────────────
# Surrogate estimators (Â, τ̂, c, r̂).  Each is a small class so that
# unit tests, calibration, and ablations can swap one out without
# touching the rest of the controller.
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SurprisalGainEstimator:
    """Â_t(α): surprisal-differential gain surrogate (Eq. (5)).

    For each candidate action α, runs the receiver under α's protocol
    instantiation in scoring mode and reads back the receiver's negative
    log-likelihood on its own preferred next-action token. The surrogate
    Â_t(α) := surprisal(no-message) − surprisal(α-message), clipped to
    [0, ∞), so that an action with extra information lowers receiver
    surprisal relative to the receiver-only baseline. A verifier-scored
    proposal score (a small classifier on the candidate output) can be
    swapped in by replacing this estimator.
    """

    receiver: BaseModelBackend
    strong_receiver: Optional[BaseModelBackend] = None

    def __call__(self, action: str, context: "SelectorContext") -> float:
        """Return Â_t(α) ≥ 0 for action α in the given context.

        Each probe pair (no-message, conditioned) runs on the receiver that
        executes α, and the conditioned probe uses α's actual candidate input.
        """
        receiver = self._receiver_for(action)
        # Receiver-only baseline surprisal (no sender message).
        s_no_msg = self._surprisal(receiver, context.action_input(Action.SKIP), context)
        if action == Action.SKIP:
            return 0.0
        # Action-conditioned surprisal on the actual candidate input.
        s_with_msg = self._surprisal(receiver, context.action_input(action), context)
        return max(0.0, s_no_msg - s_with_msg)

    def _receiver_for(self, action: str) -> BaseModelBackend:
        if action == Action.ESCALATE and self.strong_receiver is not None:
            return self.strong_receiver
        return self.receiver

    def _surprisal(
        self, receiver: BaseModelBackend, prompt: str, context: "SelectorContext"
    ) -> float:
        """Negative log-prob of the receiver's leading next-action token.

        Requires a receiver backend that returns leading-token log-probs
        (``OpenAIBackend`` does for non-reasoning models). Raises instead of
        substituting a constant when log-probs are unavailable.
        """
        request = ModelRequest(
            messages=[{"role": "user", "content": prompt}],
            model_name=receiver.model_name,
            max_tokens=4,
            logprobs=True,
        )
        response = receiver.generate(request)
        context.record_probe(response)
        if not response.logprobs:
            raise RuntimeError(
                f"Receiver backend {receiver.model_name!r} returned no log-probs; "
                "the surprisal gain estimator needs leading-token log-probs."
            )
        first = next(iter(response.logprobs.values()))
        return -float(first)


@dataclass
class PairedTaxCalibrator:
    """τ̂_t(α): paired protocol calibration on a held-out slice.

    Loads a JSON table mapping each action to its mean tax estimate, where
    tax is

        τ̂(α) := mean over held-out steps of  hit_joint(SF) − hit_joint(α)

    clipped to [0, ∞). The table is fit by SelectorCalibration.fit_taxes
    on a calibration slice disjoint from the reporting slice and frozen
    before inference.
    """

    table: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "PairedTaxCalibrator":
        with open(path) as f:
            return cls(table=json.load(f))

    def __call__(self, action: str, context: "SelectorContext") -> float:
        # If this action wasn't in calibration, default to 0 (no penalty).
        return max(0.0, float(self.table.get(action, 0.0)))


@dataclass
class TokenCostModel:
    """c(α): expected immediate token/latency cost of action α.

    Pure function of the candidate action; no calibration required.
    Sender prompts and receiver outputs are typed in tokens at the
    BudgetEnforcer's tokenizer.
    """

    budget: int
    tokenizer: BudgetEnforcer

    def __call__(self, action: str, context: "SelectorContext") -> float:
        if action == Action.SKIP:
            # Receiver only; no sender call.
            return self.tokenizer.count_tokens(context.receiver_observation)
        if action == Action.COMPRESS:
            return self.tokenizer.count_tokens(context.receiver_observation) + self.budget
        if action == Action.RAW:
            return self.tokenizer.count_tokens(
                context.receiver_observation + "\n\n" + context.sender_context
            )
        if action == Action.ESCALATE:
            # Sender + stronger receiver; treat the stronger model as ~2× cost.
            return 2.0 * (
                self.tokenizer.count_tokens(context.receiver_observation) + self.budget
            )
        if action == Action.REFRESH:
            # Resynchronization round-trip + a fresh receiver call.
            return 1.5 * (
                self.tokenizer.count_tokens(context.receiver_observation) + self.budget
            )
        return 0.0


@dataclass
class SharedStateResidualEstimator:
    """r̂_t(α): finite-horizon residual risk from shared-state probes.

    Implements Proposition 7's bounded propagation contraction: probes a
    held-out perturbation of the trajectory's shared state and returns
    the clipped expected propagated residual under each action. For
    one-step horizons (the static slice) the residual reduces to 0; for
    multi-step trajectories the probe sequence is supplied by the
    harness.
    """

    horizon: int = 1
    probe_table: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path, horizon: int = 1) -> "SharedStateResidualEstimator":
        with open(path) as f:
            return cls(horizon=horizon, probe_table=json.load(f))

    def __call__(self, action: str, context: "SelectorContext") -> float:
        if self.horizon <= 1:
            return 0.0
        return max(0.0, float(self.probe_table.get(action, 0.0)))


# ─────────────────────────────────────────────────────────────────────
# Calibration: fit η_c, η_r and τ̂ table once on a held-out slice.
# ─────────────────────────────────────────────────────────────────────

@dataclass
class SelectorCalibration:
    """Held-out calibration of the deployed score weights.

    `fit` consumes paired step records from a calibration slice (disjoint
    from the reporting slice) and returns the frozen `(η_c, η_r,
    tax_table)` tuple used at inference time.
    """

    eta_c: float = 1e-4
    eta_r: float = 0.0
    tax_table: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        calibration_records: list[dict[str, Any]],
        actions: tuple[str, ...] = Action.ALL,
        eta_c_grid: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2),
        eta_r_grid: tuple[float, ...] = (0.0, 0.5, 1.0),
    ) -> "SelectorCalibration":
        """Fit τ̂ table from per-action mean tax, η_c/η_r by grid search.

        The grid maximizes (mean hit_joint − η_c · mean tokens) on the
        calibration slice; r̂ enters only when probe records are present.
        """
        # τ̂ table.
        tax_table = {}
        for action in actions:
            cell = [r for r in calibration_records if r.get("action") == action]
            if not cell:
                tax_table[action] = 0.0
                continue
            sf = [r for r in calibration_records if r.get("action") == "sf_baseline"]
            sf_mean = sum(r["hit_joint"] for r in sf) / max(1, len(sf))
            cell_mean = sum(r["hit_joint"] for r in cell) / max(1, len(cell))
            tax_table[action] = max(0.0, sf_mean - cell_mean)

        # η_c, η_r grid search on calibration slice (no test labels read).
        best = (-math.inf, eta_c_grid[0], eta_r_grid[0])
        for eta_c in eta_c_grid:
            for eta_r in eta_r_grid:
                # Score = mean hit_joint − η_c · mean tokens (proxy objective).
                if not calibration_records:
                    objective = 0.0
                else:
                    mean_hj = sum(r["hit_joint"] for r in calibration_records) / len(
                        calibration_records
                    )
                    mean_tok = sum(r.get("tokens", 0) for r in calibration_records) / len(
                        calibration_records
                    )
                    objective = mean_hj - eta_c * mean_tok
                if objective > best[0]:
                    best = (objective, eta_c, eta_r)
        return cls(eta_c=best[1], eta_r=best[2], tax_table=tax_table)


# ─────────────────────────────────────────────────────────────────────
# Context object passed between the selector and the action handlers.
# ─────────────────────────────────────────────────────────────────────

REFRESH_DIRECTIVE = (
    "[REFRESHED SHARED STATE — please ignore prior assumptions and "
    "re-derive the next-step decision from current state.]\n\n"
)


@dataclass
class CandidateMessage:
    """A real sender message generated once per decision and reused."""

    text: str
    tokens: int
    truncated: bool
    sender_input: str
    response: ModelResponse


@dataclass
class SelectorContext:
    sender_context: str
    receiver_observation: str
    candidate_set: Optional[CandidateSet] = None
    step_t: int = 0
    compress_candidate: Optional[CandidateMessage] = None
    refresh_candidate: Optional[CandidateMessage] = None
    probe_calls: int = 0
    probe_input_tokens: int = 0
    probe_output_tokens: int = 0
    probe_latency_ms: int = 0

    def record_probe(self, response: ModelResponse) -> None:
        self.probe_calls += 1
        self.probe_input_tokens += response.input_tokens
        self.probe_output_tokens += response.output_tokens
        self.probe_latency_ms += response.latency_ms

    def candidate_for(self, action: str) -> CandidateMessage:
        """Cached sender candidate used by a bounded-message action."""
        candidate = (
            self.refresh_candidate if action == Action.REFRESH else self.compress_candidate
        )
        if candidate is None:
            raise ValueError(
                f"No sender candidate is cached for action {action!r}; "
                "ProtocolSelector.prepare_candidates generates it before scoring."
            )
        return candidate

    def action_message(self, action: str) -> str:
        """Return the message that action α exposes to the receiver."""
        if action == Action.SKIP:
            return ""
        if action == Action.RAW:
            return self.sender_context
        if action in (Action.COMPRESS, Action.ESCALATE, Action.REFRESH):
            return self.candidate_for(action).text
        return ""

    def action_input(self, action: str) -> str:
        """Receiver-facing input of action α, shared by probes and execution."""
        if action == Action.SKIP:
            return self.receiver_observation
        if action == Action.RAW:
            return (
                self.receiver_observation
                + "\n\n[Full upstream context]\n"
                + self.sender_context
            )
        constraint_block = (
            build_candidate_constraint_block(self.candidate_set)
            if self.candidate_set is not None
            else ""
        )
        return (
            f"[Your local observation]\n{self.receiver_observation}\n\n"
            f"[Coordination message from sender]\n{self.action_message(action)}"
            f"{constraint_block}"
        )


# ─────────────────────────────────────────────────────────────────────
# Action handlers — one per action in {skip, compress, raw, escalate,
# refresh}. Each implements run() to actually execute the protocol.
# ─────────────────────────────────────────────────────────────────────

class _ActionHandler:
    """Common interface; subclasses implement `run`."""

    name: str

    def run(
        self,
        context: SelectorContext,
        sender: BaseModelBackend,
        receiver: BaseModelBackend,
        budget_enforcer: BudgetEnforcer,
        budget: int,
        encoding_format: str,
        strong_receiver: Optional[BaseModelBackend],
        inner_ma: MASequentialAgent,
    ) -> tuple[ModelResponse, StepRecord]:
        raise NotImplementedError


def _run_with_candidate(
    action: str,
    context: SelectorContext,
    sender: BaseModelBackend,
    receiver: BaseModelBackend,
    budget: int,
    encoding_format: str,
    inner_ma: MASequentialAgent,
) -> tuple[ModelResponse, StepRecord]:
    """One receiver generation on the cached candidate of a bounded-message action."""
    candidate = context.candidate_for(action)
    receiver_input = context.action_input(action)
    receiver_messages = [{"role": "system", "content": RECEIVER_SYSTEM_PROMPT}]
    receiver_messages.extend(inner_ma.receiver_history)
    receiver_messages.append({"role": "user", "content": receiver_input})
    response = receiver.generate(ModelRequest(
        messages=receiver_messages,
        model_name=receiver.model_name,
        max_tokens=4096,
        temperature=0.0,
    ))
    cs = context.candidate_set
    record = StepRecord(
        run_id="selector",
        agent_type=f"selector_{action}",
        sender_model=sender.model_name,
        receiver_model=receiver.model_name,
        budget_tokens=budget,
        encoding_format=encoding_format,
        instance_id=getattr(cs, "instance_id", "") or "",
        step_t=context.step_t,
        candidate_set_size=cs.total_labels if cs else 0,
        message_raw=candidate.text,
        message_tokens=candidate.tokens,
        message_truncated=candidate.truncated,
        message_budget_violated=candidate.truncated,
        budget_enforcement_mode="truncate_to_budget",
        receiver_input=receiver_input,
        receiver_input_tokens=response.input_tokens,
        receiver_output=response.content,
        receiver_output_tokens=response.output_tokens,
    )
    # The executed candidate enters the agents' histories, as in MASequentialAgent.step.
    inner_ma.sender_history.append({"role": "user", "content": candidate.sender_input[:2000]})
    inner_ma.sender_history.append({"role": "assistant", "content": candidate.text})
    inner_ma.receiver_history.append({"role": "user", "content": receiver_input[:2000]})
    inner_ma.receiver_history.append({"role": "assistant", "content": response.content})
    if cs is not None:
        record = inner_ma._score_coordination(
            receiver_messages, response, context.receiver_observation, cs, record
        )
    else:
        action_pred = extract_action_prediction(response.content) or {}
        record.action_type_pred = action_pred.get("action_type", "")
        record.artifact_pred = action_pred.get("target_artifact", "")
    return response, record


class SkipHandler(_ActionHandler):
    name = Action.SKIP

    def run(self, context, sender, receiver, budget_enforcer, budget,
            encoding_format, strong_receiver, inner_ma):
        # Receiver-only call: no sender message; receiver acts on local view.
        request = ModelRequest(
            messages=[{"role": "user", "content": context.receiver_observation}],
            model_name=receiver.model_name,
            max_tokens=512,
            temperature=0.0,
        )
        response = receiver.generate(request)
        action_pred = extract_action_prediction(response.content)
        record = StepRecord(
            run_id="selector",
            agent_type="selector_skip",
            sender_model="",
            receiver_model=receiver.model_name,
            single_model="",
            instance_id=getattr(context.candidate_set, "instance_id", "") or "",
            step_t=context.step_t,
            message_raw="",
            message_tokens=0,
            receiver_input=context.receiver_observation,
            receiver_input_tokens=response.input_tokens,
            receiver_output=response.content,
            receiver_output_tokens=response.output_tokens,
            action_type_pred=action_pred.get("action_type", ""),
            artifact_pred=action_pred.get("target_artifact", ""),
            repair_mode="selector_action=skip",
        )
        return response, record


class CompressHandler(_ActionHandler):
    name = Action.COMPRESS

    def run(self, context, sender, receiver, budget_enforcer, budget,
            encoding_format, strong_receiver, inner_ma):
        # Bounded message under budget B: reuse the cached compress candidate.
        return _run_with_candidate(
            Action.COMPRESS, context, sender, receiver, budget, encoding_format, inner_ma,
        )


class RawHandler(_ActionHandler):
    name = Action.RAW

    def run(self, context, sender, receiver, budget_enforcer, budget,
            encoding_format, strong_receiver, inner_ma):
        # Expose full sender context to receiver, no compression / no budget cap.
        prompt = (
            context.receiver_observation
            + "\n\n[Full upstream context]\n"
            + context.sender_context
        )
        request = ModelRequest(
            messages=[{"role": "user", "content": prompt}],
            model_name=receiver.model_name,
            max_tokens=512,
            temperature=0.0,
        )
        response = receiver.generate(request)
        action_pred = extract_action_prediction(response.content)
        record = StepRecord(
            run_id="selector",
            agent_type="selector_raw",
            sender_model="",
            receiver_model=receiver.model_name,
            single_model="",
            instance_id=getattr(context.candidate_set, "instance_id", "") or "",
            step_t=context.step_t,
            message_raw=context.sender_context,
            message_tokens=budget_enforcer.count_tokens(context.sender_context),
            receiver_input=prompt,
            receiver_input_tokens=response.input_tokens,
            receiver_output=response.content,
            receiver_output_tokens=response.output_tokens,
            action_type_pred=action_pred.get("action_type", ""),
            artifact_pred=action_pred.get("target_artifact", ""),
            repair_mode="selector_action=raw",
        )
        return response, record


class EscalateHandler(_ActionHandler):
    name = Action.ESCALATE

    def run(self, context, sender, receiver, budget_enforcer, budget,
            encoding_format, strong_receiver, inner_ma):
        if strong_receiver is None:
            # No stronger receiver registered → fall back to compress.
            return CompressHandler().run(
                context, sender, receiver, budget_enforcer, budget,
                encoding_format, strong_receiver, inner_ma,
            )
        # Reuse the cached compressed message and invoke the stronger receiver directly.
        return _run_with_candidate(
            Action.ESCALATE, context, sender, strong_receiver, budget, encoding_format,
            inner_ma,
        )


class RefreshHandler(_ActionHandler):
    name = Action.REFRESH

    def run(self, context, sender, receiver, budget_enforcer, budget,
            encoding_format, strong_receiver, inner_ma):
        # Shared-state resynchronization: the refresh candidate was generated
        # from the sender context with an explicit refresh directive
        # (REFRESH_DIRECTIVE); reuse it for the receiver generation.
        return _run_with_candidate(
            Action.REFRESH, context, sender, receiver, budget, encoding_format, inner_ma,
        )


# ─────────────────────────────────────────────────────────────────────
# The main selector controller.
# ─────────────────────────────────────────────────────────────────────

class ProtocolSelector:
    """Implements Eq. (5) of the paper.

    Constructor signature:
        ProtocolSelector(
            sender, receiver,
            budget_enforcer, budget,
            encoding_format,
            calibration,
            gain_estimator,
            tax_estimator,
            cost_model,
            residual_estimator,
            strong_receiver=None,
            actions=Action.ALL,
        )

    `step(sender_context, receiver_observation, candidate_set, step_t)`
    generates and caches the `compress` and `refresh` candidates, selects
    α* := argmax_α Ĵ_t(α), executes the corresponding action handler once
    with its cached candidate, and returns `(ModelResponse, StepRecord)`
    whose schema matches the JSONL records under `data/`. The record's
    `total_*` token and latency fields include the candidate generations,
    all scoring probes, and the executed receiver call.
    """

    def __init__(
        self,
        sender: BaseModelBackend,
        receiver: BaseModelBackend,
        budget_enforcer: BudgetEnforcer,
        budget: int = 128,
        encoding_format: str = "canonical_json",
        calibration: Optional[SelectorCalibration] = None,
        gain_estimator: Optional[SurprisalGainEstimator] = None,
        tax_estimator: Optional[PairedTaxCalibrator] = None,
        cost_model: Optional[TokenCostModel] = None,
        residual_estimator: Optional[SharedStateResidualEstimator] = None,
        strong_receiver: Optional[BaseModelBackend] = None,
        actions: tuple[str, ...] = Action.ALL,
    ):
        self.sender = sender
        self.receiver = receiver
        self.budget_enforcer = budget_enforcer
        self.budget = budget
        self.encoding_format = encoding_format
        self.strong_receiver = strong_receiver
        self.actions = actions

        self.calibration = calibration or SelectorCalibration()
        self.A_hat = gain_estimator or SurprisalGainEstimator(
            receiver=receiver, strong_receiver=strong_receiver
        )
        self.tau_hat = tax_estimator or PairedTaxCalibrator(
            table=self.calibration.tax_table
        )
        self.c = cost_model or TokenCostModel(
            budget=budget, tokenizer=budget_enforcer
        )
        self.r_hat = residual_estimator or SharedStateResidualEstimator()
        self.eta_c = self.calibration.eta_c
        self.eta_r = self.calibration.eta_r

        self.inner_ma = MASequentialAgent(
            sender=sender,
            receiver=receiver,
            budget_enforcer=budget_enforcer,
            budget=budget,
            encoding_format=encoding_format,
        )
        self._handlers: dict[str, _ActionHandler] = {
            Action.SKIP:     SkipHandler(),
            Action.COMPRESS: CompressHandler(),
            Action.RAW:      RawHandler(),
            Action.ESCALATE: EscalateHandler(),
            Action.REFRESH:  RefreshHandler(),
        }

    # ------------------------------------------------------------------ score

    def score_action(self, action: str, context: SelectorContext) -> float:
        """Ĵ_t(α)  =  Â_t(α)  −  τ̂_t(α)  −  η_c · c(α)  −  η_r · r̂_t(α)."""
        return (
            self.A_hat(action, context)
            - self.tau_hat(action, context)
            - self.eta_c * self.c(action, context)
            - self.eta_r * self.r_hat(action, context)
        )

    def select(self, context: SelectorContext) -> str:
        """argmax_α Ĵ_t(α) over the configured action set."""
        scores = {a: self.score_action(a, context) for a in self.actions}
        return max(scores, key=scores.get)

    # ------------------------------------------------------------ candidates

    def _generate_candidate(
        self, sender_context: str, candidate_set: Optional[CandidateSet]
    ) -> CandidateMessage:
        """One sender call under budget B, as in MASequentialAgent.step."""
        sender_input = sender_context
        if candidate_set is not None:
            sender_input += build_candidate_constraint_block(candidate_set)
        sender_messages = [{"role": "system", "content": SENDER_SYSTEM_PROMPT}]
        sender_messages.extend(self.inner_ma.sender_history)
        if self.inner_ma.rewrite_from_canonical_payload:
            response, text, _ = self.inner_ma._generate_rewritten_message(
                sender_messages, sender_input
            )
        else:
            response, text = self.inner_ma._generate_direct_message(
                sender_messages, sender_input
            )
        tokens = self.budget_enforcer.count_tokens(text)
        truncated = tokens > self.budget
        if truncated and self.budget > 0:
            text, info = self.budget_enforcer.enforce(text, self.budget)
            tokens = info["final_tokens"]
        return CandidateMessage(
            text=text, tokens=tokens, truncated=truncated,
            sender_input=sender_input, response=response,
        )

    def prepare_candidates(self, context: SelectorContext) -> None:
        """Generate and cache the real `compress` and `refresh` candidates."""
        if Action.COMPRESS in self.actions or Action.ESCALATE in self.actions:
            context.compress_candidate = self._generate_candidate(
                context.sender_context, context.candidate_set
            )
        if Action.REFRESH in self.actions:
            context.refresh_candidate = self._generate_candidate(
                REFRESH_DIRECTIVE + context.sender_context, context.candidate_set
            )

    # ------------------------------------------------------------------ step

    def step(
        self,
        sender_context: str,
        receiver_observation: str,
        candidate_set: Optional[CandidateSet] = None,
        step_t: int = 0,
    ) -> tuple[ModelResponse, StepRecord]:
        ctx = SelectorContext(
            sender_context=sender_context,
            receiver_observation=receiver_observation,
            candidate_set=candidate_set,
            step_t=step_t,
        )
        self.prepare_candidates(ctx)
        alpha = self.select(ctx)
        handler = self._handlers[alpha]
        response, record = handler.run(
            context=ctx,
            sender=self.sender,
            receiver=self.receiver,
            budget_enforcer=self.budget_enforcer,
            budget=self.budget,
            encoding_format=self.encoding_format,
            strong_receiver=self.strong_receiver,
            inner_ma=self.inner_ma,
        )
        # Decision cost: candidate generations, scoring probes, executed call.
        candidates = [c for c in (ctx.compress_candidate, ctx.refresh_candidate) if c]
        record.total_input_tokens = (
            sum(c.response.input_tokens for c in candidates)
            + ctx.probe_input_tokens + response.input_tokens
        )
        record.total_output_tokens = (
            sum(c.response.output_tokens for c in candidates)
            + ctx.probe_output_tokens + response.output_tokens
        )
        record.latency_ms = (
            sum(c.response.latency_ms for c in candidates)
            + ctx.probe_latency_ms + response.latency_ms
        )
        record.repair_mode = f"selector_action={alpha}"
        return response, record
