"""
agents/ma_sequential.py — Multi-Agent Sequential Delegation.

The canonical sender-receiver paradigm from the theoretical framework:
  Sender (richer context) → encodes M_t (≤ B_t tokens) → Receiver (executes action)

This is the PRIMARY multi-agent paradigm tested in the paper.
"""

from __future__ import annotations
from typing import Optional
import logging
import json

from core.types import (
    ModelRequest, ModelResponse, StepRecord, CandidateSet, EncodingFormat
)
from core.model_interface import BaseModelBackend
from core.tokenizer import BudgetEnforcer, EncodingFormatter
from core.output_parsing import extract_action_prediction, extract_json_object

logger = logging.getLogger(__name__)


SENDER_SYSTEM_PROMPT = """You are the SENDER agent in a multi-agent coordination system.
You have access to the full task context (repository state, failing tests, plans).
Your job: encode the most important coordination information into a concise message
for the RECEIVER agent, who will execute the next action.

The receiver has LIMITED context — they can only see their local observation
plus YOUR message. You must convey:
1. What action to take (action_type)
2. Which artifact to target (file, test, config)
3. Key context the receiver needs to act correctly
4. Any critical constraints or dependencies

Respond with your coordination message. Be precise and information-dense.
Format your message as specified by the encoding format instruction."""

RECEIVER_SYSTEM_PROMPT = """You are the RECEIVER agent in a multi-agent coordination system.
You receive a coordination message from the SENDER agent, plus your own local observation.
Your job: decode the message and execute the correct action.

Respond in this JSON format:
{
  "action_type": "<one of: edit, create, delete, run_test, read, configure>",
  "target_artifact": "<file path, test name, or config entry>",
  "rationale": "<brief explanation of why this action>",
  "content": "<actual code/command if applicable>"
}"""


ENCODING_INSTRUCTIONS = {
    "prose": "Write your coordination message as natural prose paragraphs.",
    "bullet_schema": "Write your coordination message as structured bullet points:\n- action: ...\n- target: ...\n- rationale: ...\n- context: ...",
    "canonical_json": 'Write your coordination message as a JSON object with keys: "action_type", "target_artifact", "rationale", "context", "constraints".',
    "decision_header_json": (
        'Write a two-part message. First line: `DECISION action_type=<...> target_artifact=<...>`. '
        'Then output a compact JSON object with keys "action_type", "target_artifact", "rationale", "context", "constraints".'
    ),
    "bullet_json_footer": (
        'Write 2-4 short bullet lines for action_type, target_artifact, rationale, constraints. '
        'Then output `JSON_FOOTER` followed by a compact JSON object with keys '
        '"action_type", "target_artifact", "rationale", "context", "constraints".'
    ),
}

CANONICAL_JSON_ONLY_INSTRUCTION = """Return exactly one valid JSON object.
Do not include markdown, code fences, analysis, commentary, or surrounding text.
Use these keys exactly: "action_type", "target_artifact", "rationale", "context", "constraints"."""


class MASequentialAgent:
    """
    Multi-Agent Sequential Delegation.
    
    Flow per step:
      1. Sender sees full context → produces message M_t (≤ B_t tokens)
      2. Budget enforcer truncates M_t if needed
      3. Receiver sees local observation + M_t → produces action
      
    Key parameters:
      - sender_model: the model with richer context
      - receiver_model: the model that executes (can be weaker)
      - budget: token limit on M_t
      - encoding_format: how sender structures the message
    """

    def __init__(
        self,
        sender: BaseModelBackend,
        receiver: BaseModelBackend,
        budget_enforcer: BudgetEnforcer,
        budget: int = 512,
        encoding_format: str = "canonical_json",
        rewrite_from_canonical_payload: bool = False,
    ):
        self.sender = sender
        self.receiver = receiver
        self.budget_enforcer = budget_enforcer
        self.budget = budget
        self.encoding_format = encoding_format
        self.rewrite_from_canonical_payload = rewrite_from_canonical_payload

        self.sender_history: list[dict] = []
        self.receiver_history: list[dict] = []

    def reset(self):
        self.sender_history = []
        self.receiver_history = []

    def set_budget(self, budget: int):
        """Update budget (used by budget_sweep protocol)."""
        self.budget = budget

    def set_encoding_format(self, fmt: str):
        """Update encoding format (used by encoding_rewrite protocol)."""
        self.encoding_format = fmt

    def swap_receiver(self, new_receiver: BaseModelBackend):
        """Swap receiver model (used by receiver_swap protocol)."""
        self.receiver = new_receiver

    def step(
        self,
        sender_context: str,
        receiver_observation: str,
        candidate_set: Optional[CandidateSet] = None,
        step_t: int = 0,
    ) -> tuple[ModelResponse, StepRecord]:
        """
        Execute one coordination step.
        
        Args:
            sender_context: full context visible to sender
            receiver_observation: local observation visible to receiver
            candidate_set: optional Z_t for CSI measurement
            step_t: step index
            
        Returns:
            (receiver_response, step_record)
        """
        record = StepRecord(
            agent_type="ma_sequential",
            sender_model=self.sender.model_name,
            receiver_model=self.receiver.model_name,
            budget_tokens=self.budget,
            encoding_format=self.encoding_format,
            step_t=step_t,
            candidate_set_size=candidate_set.total_labels if candidate_set else 0,
        )

        # ── Step 1: Sender encodes message ──
        sender_messages = [
            {"role": "system", "content": SENDER_SYSTEM_PROMPT},
        ]
        for h in self.sender_history:
            sender_messages.append(h)

        # Inject candidate constraint (shared with single_full and receiver_only)
        if candidate_set is not None:
            from core.constraint_injection import build_candidate_constraint_block
            sender_context += build_candidate_constraint_block(candidate_set)

        if self.rewrite_from_canonical_payload:
            sender_response, raw_message, rewrite_meta = self._generate_rewritten_message(
                sender_messages,
                sender_context,
            )
        else:
            sender_response, raw_message = self._generate_direct_message(
                sender_messages,
                sender_context,
            )
            rewrite_meta = {}

        # ── Step 2: Budget accounting + enforcement ──
        raw_message_tokens = self.budget_enforcer.count_tokens(raw_message)
        message_budget_violated = raw_message_tokens > self.budget

        # Actually enforce budget: truncate message to budget tokens
        if message_budget_violated and self.budget > 0:
            raw_message, _trunc_info = self.budget_enforcer.enforce(
                raw_message, self.budget
            )
            raw_message_tokens = _trunc_info["final_tokens"]

        record.message_raw = raw_message
        record.message_tokens = raw_message_tokens
        record.message_truncated = message_budget_violated
        record.budget_enforcement_mode = "truncate_to_budget"
        record.message_budget_violated = message_budget_violated
        record.canonical_payload_extracted = rewrite_meta.get(
            "canonical_payload_extracted", False
        )
        record.canonical_payload_source = rewrite_meta.get(
            "canonical_payload_source", ""
        )
        record.canonical_payload_tokens = rewrite_meta.get(
            "canonical_payload_tokens", 0
        )

        # Update sender history
        self.sender_history.append({"role": "user", "content": sender_context[:2000]})  # truncate history
        self.sender_history.append({"role": "assistant", "content": raw_message})

        # ── Step 3: Receiver decodes and acts ──
        receiver_messages = [
            {"role": "system", "content": RECEIVER_SYSTEM_PROMPT},
        ]
        for h in self.receiver_history:
            receiver_messages.append(h)

        # Inject candidate constraint to receiver too (fairness: all agents see it)
        constraint_block = ""
        if candidate_set is not None:
            from core.constraint_injection import build_candidate_constraint_block
            constraint_block = build_candidate_constraint_block(candidate_set)

        receiver_input = (
            f"[Your local observation]\n{receiver_observation}\n\n"
            f"[Coordination message from sender]\n{raw_message}"
            f"{constraint_block}"
        )
        receiver_messages.append({"role": "user", "content": receiver_input})

        receiver_request = ModelRequest(
            messages=receiver_messages,
            model_name=self.receiver.model_name,
            max_tokens=4096,
            temperature=0.0,
            logprobs=candidate_set is not None,
        )
        receiver_response = self.receiver.generate(receiver_request)

        record.receiver_input_tokens = receiver_response.input_tokens
        record.receiver_input = receiver_input
        record.receiver_output = receiver_response.content
        record.receiver_output_tokens = receiver_response.output_tokens
        record.total_input_tokens = sender_response.input_tokens + receiver_response.input_tokens
        record.total_output_tokens = sender_response.output_tokens + receiver_response.output_tokens
        record.latency_ms = sender_response.latency_ms + receiver_response.latency_ms

        # Update receiver history
        self.receiver_history.append({"role": "user", "content": receiver_input[:2000]})
        self.receiver_history.append({"role": "assistant", "content": receiver_response.content})

        if message_budget_violated:
            logger.warning(
                "Sender message exceeded budget after generation: model=%s budget=%s actual_tokens=%s",
                self.sender.model_name,
                self.budget,
                raw_message_tokens,
            )

        # ── Step 4: Score coordination target ──
        if candidate_set is not None:
            record = self._score_coordination(
                receiver_messages, receiver_response,
                receiver_observation, candidate_set, record
            )

        return receiver_response, record

    def _generate_direct_message(
        self,
        sender_messages: list[dict],
        sender_context: str,
    ) -> tuple[ModelResponse, str]:
        encoding_inst = ENCODING_INSTRUCTIONS.get(self.encoding_format, "")
        sender_messages = list(sender_messages)
        sender_messages.append({
            "role": "user",
            "content": f"{sender_context}\n\n[Encoding instruction: {encoding_inst}]\n\n"
                       f"[Budget: Your message must fit in {self.budget} tokens. "
                       f"Be maximally information-dense.]"
        })

        sender_request = ModelRequest(
            messages=sender_messages,
            model_name=self.sender.model_name,
            max_tokens=min(self.budget, 4096),
            temperature=0.0,
        )
        sender_response = self.sender.generate(sender_request)
        return sender_response, sender_response.content

    def _generate_rewritten_message(
        self,
        sender_messages: list[dict],
        sender_context: str,
    ) -> tuple[ModelResponse, str, dict]:
        sender_messages = list(sender_messages)
        sender_messages.append({
            "role": "user",
            "content": (
                f"{sender_context}\n\n"
                f"[Canonical payload instruction]\n{CANONICAL_JSON_ONLY_INSTRUCTION}\n\n"
                f"[Budget note: encode the key coordination semantics compactly enough that the final "
                f"rewritten message can fit in {self.budget} tokens.]"
            )
        })

        canonical_token_budget = max(256, min(max(self.budget * 2, self.budget), 1024))
        sender_request = ModelRequest(
            messages=sender_messages,
            model_name=self.sender.model_name,
            max_tokens=canonical_token_budget,
            temperature=0.0,
        )
        sender_response = self.sender.generate(sender_request)

        payload = extract_json_object(
            sender_response.content,
            preferred_keys=(
                "action_type",
                "target_artifact",
                "rationale",
                "context",
                "constraints",
            ),
        )
        payload_source = "direct"
        if payload is None:
            repair_messages = list(sender_messages)
            repair_messages.append({"role": "assistant", "content": sender_response.content})
            repair_messages.append({
                "role": "user",
                "content": CANONICAL_JSON_ONLY_INSTRUCTION,
            })
            repair_request = ModelRequest(
                messages=repair_messages,
                model_name=self.sender.model_name,
                max_tokens=canonical_token_budget,
                temperature=0.0,
            )
            repair_response = self.sender.generate(repair_request)
            payload = extract_json_object(
                repair_response.content,
                preferred_keys=(
                    "action_type",
                    "target_artifact",
                    "rationale",
                    "context",
                    "constraints",
                ),
            )
            sender_response = ModelResponse(
                content=repair_response.content,
                input_tokens=sender_response.input_tokens + repair_response.input_tokens,
                output_tokens=sender_response.output_tokens + repair_response.output_tokens,
                logprobs=repair_response.logprobs,
                latency_ms=sender_response.latency_ms + repair_response.latency_ms,
                model_name=repair_response.model_name or sender_response.model_name,
                raw_response=repair_response.raw_response,
            )
            payload_source = "repair_retry"

        if payload is None:
            logger.warning(
                "Failed to extract canonical payload for encoding rewrite; falling back to raw sender output."
            )
            return sender_response, sender_response.content, {
                "canonical_payload_extracted": False,
                "canonical_payload_source": "fallback_raw",
                "canonical_payload_tokens": 0,
            }

        try:
            rewritten = EncodingFormatter.format(payload, self.encoding_format)
        except Exception as exc:
            logger.warning(
                "Failed to format canonical payload into %s: %s; falling back to raw sender output.",
                self.encoding_format,
                exc,
            )
            return sender_response, sender_response.content, {
                "canonical_payload_extracted": True,
                "canonical_payload_source": payload_source,
                "canonical_payload_tokens": self.budget_enforcer.count_tokens(
                    EncodingFormatter.to_canonical_json(payload)
                ),
            }

        return sender_response, rewritten, {
            "canonical_payload_extracted": True,
            "canonical_payload_source": payload_source,
            "canonical_payload_tokens": self.budget_enforcer.count_tokens(
                EncodingFormatter.to_canonical_json(payload)
            ),
        }

    def _score_coordination(
        self,
        receiver_messages: list[dict],
        response: ModelResponse,
        receiver_observation: str,
        cs: CandidateSet,
        record: StepRecord,
        receiver_backend: BaseModelBackend | None = None,
    ) -> StepRecord:
        """Score prediction against Z_t and compute CSI components."""
        import math
        import time as _time
        scorer = receiver_backend or self.receiver

        # Parse prediction
        pred = extract_action_prediction(response.content)
        if pred is not None:
            record.action_type_pred = pred.get("action_type", "")
            record.artifact_pred = pred.get("target_artifact", "")
        else:
            record.action_type_pred = ""
            record.artifact_pred = ""

        # Near-miss correction (shared with single_full and receiver_only)
        from core.near_miss import apply_near_miss_correction
        if record.artifact_pred and record.artifact_pred not in cs.artifacts:
            record.artifact_pred, _ = apply_near_miss_correction(
                record.artifact_pred, cs.artifacts
            )
        if record.action_type_pred and record.action_type_pred not in cs.action_types:
            record.action_type_pred, _ = apply_near_miss_correction(
                record.action_type_pred, cs.action_types
            )

        record.action_type_gold = cs.oracle_action
        record.artifact_gold = cs.oracle_artifact
        record.hit_action = record.action_type_pred == cs.oracle_action
        record.hit_artifact = record.artifact_pred == cs.oracle_artifact
        record.hit_joint = record.hit_action and record.hit_artifact

        # ── CSI computation ──
        # Build candidate strings for all joint labels
        candidate_strings = []
        for act in cs.action_types:
            for art in cs.artifacts:
                candidate_strings.append(
                    f'{{"action_type": "{act}", "target_artifact": "{art}"}}'
                )

        oracle_idx = cs.joint_label(cs.oracle_action, cs.oracle_artifact)
        uniform_surprisal = math.log2(cs.total_labels)

        # Determine scoring path: exact only if backend supports it
        use_exact = (
            hasattr(scorer, 'scoring_mode')
            and scorer.scoring_mode() == "exact"
        )

        scored_exact = False
        if use_exact:
            try:
                t0 = _time.perf_counter()
                scores_with_msg = scorer.score_candidates(receiver_messages, candidate_strings)
                scoring_latency = int((_time.perf_counter() - t0) * 1000)

                oracle_lp = scores_with_msg.get(oracle_idx, -100.0)

                # Guard: if oracle logprob is suspiciously low, exact scoring failed
                if oracle_lp <= -99.0:
                    raise RuntimeError(
                        f"Oracle candidate {oracle_idx} not reliably scored (lp={oracle_lp:.1f}); "
                        f"falling back to proxy"
                    )

                record.surprisal_bits = -oracle_lp / math.log(2)
                record.candidate_logprobs = {str(k): v for k, v in scores_with_msg.items()}
                record.scoring_latency_ms = scoring_latency
                record.scoring_mode = "exact"

                # Score WITHOUT message (no-message baseline)
                no_msg_surprisal = self.compute_no_message_baseline(
                    receiver_observation,
                    cs,
                    receiver_backend=scorer,
                )
                record.surprisal_no_msg = no_msg_surprisal
                record.csi_sample = record.surprisal_no_msg - record.surprisal_bits
                scored_exact = True

            except Exception as e:
                logger.debug(f"Exact CSI scoring failed, using proxy: {e}")

        # Proxy fallback: hit-based discrete CSI
        if not scored_exact:
            if record.hit_joint:
                record.surprisal_bits = 0.5
            elif record.hit_action or record.hit_artifact:
                record.surprisal_bits = 3.0
            else:
                record.surprisal_bits = 6.0

            # Compute proxy no-message baseline by running receiver without message
            try:
                no_msg_surp = self._proxy_no_message_baseline(
                    receiver_observation, cs, receiver_backend=scorer,
                )
                record.surprisal_no_msg = no_msg_surp
            except Exception as _exc:
                logger.debug("Proxy no-message baseline failed: %s", _exc)
                record.surprisal_no_msg = uniform_surprisal

            record.csi_sample = record.surprisal_no_msg - record.surprisal_bits
            record.scoring_mode = "proxy"

        return record

    def _proxy_no_message_baseline(
        self,
        receiver_observation: str,
        candidate_set: CandidateSet,
        receiver_backend: BaseModelBackend | None = None,
    ) -> float:
        """
        Proxy no-message baseline: run receiver without sender message,
        extract prediction, score with same hit-based proxy as with-message.
        Returns surprisal in bits (0.5, 3.0, or 6.0).
        """
        messages = [
            {"role": "system", "content": RECEIVER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[Your local observation]\n{receiver_observation}\n\n"
                f"[No coordination message available. Use only your local observation.]"
            )},
        ]
        backend = receiver_backend or self.receiver
        from core.types import ModelRequest
        req = ModelRequest(
            messages=messages,
            model_name=backend.model_name,
            max_tokens=256,
            temperature=0.0,
        )
        resp = backend.generate(req)
        pred = extract_action_prediction(resp.content)

        if pred is not None:
            hit_action = pred.get("action_type", "") == candidate_set.oracle_action
            hit_artifact = pred.get("target_artifact", "") == candidate_set.oracle_artifact
            hit_joint = hit_action and hit_artifact
        else:
            hit_action = False
            hit_artifact = False
            hit_joint = False

        if hit_joint:
            return 0.5
        elif hit_action or hit_artifact:
            return 3.0
        else:
            return 6.0

    def compute_no_message_baseline(
        self,
        receiver_observation: str,
        candidate_set: CandidateSet,
        receiver_backend: BaseModelBackend | None = None,
    ) -> float:
        """
        Compute H_C(Z|X) — receiver's uncertainty WITHOUT the sender's message.
        This is the critical baseline for CSI computation.
        
        Returns: surprisal in bits for the oracle label
        """
        messages = [
            {"role": "system", "content": RECEIVER_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"[Your local observation]\n{receiver_observation}\n\n"
                f"[No coordination message available. Use only your local observation.]"
            )},
        ]
        scorer = receiver_backend or self.receiver

        # Build candidate strings for scoring
        candidates = []
        for act in candidate_set.action_types:
            for art in candidate_set.artifacts:
                candidates.append(f'{{"action_type": "{act}", "target_artifact": "{art}"}}')

        # Score all candidates
        scores = scorer.score_candidates(messages, candidates)

        # Get oracle label index
        oracle_idx = candidate_set.joint_label(
            candidate_set.oracle_action, candidate_set.oracle_artifact
        )

        import math
        oracle_logprob = scores.get(oracle_idx, -100.0)
        # Convert log_e to log_2
        surprisal_bits = -oracle_logprob / math.log(2)
        return surprisal_bits
