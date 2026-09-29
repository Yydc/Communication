"""
core/types.py — Shared data structures for the MA-LLM evaluation framework.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import time
import json


# ── Enums ─────────────────────────────────────────────

class AgentType(str, Enum):
    SINGLE = "single"
    SINGLE_FULL = "single_full"
    RECEIVER_ONLY = "receiver_only"
    MA_SEQUENTIAL = "ma_sequential"
    MA_ITERATIVE = "ma_iterative"
    MA_PARALLEL = "ma_parallel"
    SELECTOR_ALWAYS_SEND = "selector_always_send"
    SELECTOR_ALWAYS_SKIP = "selector_always_skip"
    SELECTOR_ALWAYS_REWRITE = "selector_always_rewrite"
    SELECTOR_ADAPTIVE = "selector_adaptive"


class ProtocolType(str, Enum):
    BUDGET_SWEEP = "budget_sweep"
    RECEIVER_SWAP = "receiver_swap"
    PERTURB_ROLLOUT = "perturb_rollout"
    ENCODING_REWRITE = "encoding_rewrite"
    SPARSE_REPAIR = "sparse_repair"


class BenchmarkFamily(str, Enum):
    SHARED_CONTEXT = "shared_context"
    SEARCH_LIKE = "search_like"
    DUAL_CONTROL = "dual_control"
    DIAGNOSIS = "diagnosis"
    CONTROL = "control"
    TOOL_USE = "tool_use"


class EncodingFormat(str, Enum):
    PROSE = "prose"
    BULLET_SCHEMA = "bullet_schema"
    CANONICAL_JSON = "canonical_json"
    DECISION_HEADER_JSON = "decision_header_json"
    BULLET_JSON_FOOTER = "bullet_json_footer"


class ModelBackend(str, Enum):
    LOCAL = "local"
    API = "api"


# ── Model I/O ─────────────────────────────────────────

@dataclass
class ModelRequest:
    """A single request to an LLM."""
    messages: list[dict]          # [{"role": "system"|"user"|"assistant", "content": ...}]
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    logprobs: bool = False        # request log-probs over candidate tokens
    logprob_tokens: list[str] | None = None  # specific tokens to score
    stop: list[str] | None = None


@dataclass
class ModelResponse:
    """Response from an LLM call."""
    content: str
    input_tokens: int
    output_tokens: int
    logprobs: dict | None = None  # token → log-prob mapping
    latency_ms: int = 0
    model_name: str = ""
    raw_response: dict | None = None


# ── Candidate Set & Coordination Target ───────────────

@dataclass
class CandidateSet:
    """The Z_t candidate set for coordination target measurement."""
    instance_id: str
    step_t: int
    action_types: list[str]       # |T_act| = 6
    artifacts: list[str]          # |A_t| = 32
    oracle_action: str            # gold action type
    oracle_artifact: str          # gold artifact
    oracle_index: int             # index in artifacts list

    @property
    def total_labels(self) -> int:
        return len(self.action_types) * len(self.artifacts)

    def joint_label(self, action: str, artifact: str) -> int:
        """Return flat index for (action, artifact) pair."""
        a_idx = self.action_types.index(action)
        art_idx = self.artifacts.index(artifact)
        return a_idx * len(self.artifacts) + art_idx


# ── Per-Step Log Entry ────────────────────────────────

@dataclass
class StepRecord:
    """Everything we log for one agent step. This is the fundamental data unit."""

    # ── Run-level
    run_id: str = ""
    protocol: str = ""
    benchmark: str = ""           # benchmark name (e.g. "swe_bench_pro")
    benchmark_family: str = ""    # family (e.g. "shared_context")
    agent_type: str = ""
    sender_model: str = ""
    receiver_model: str = ""
    single_model: str = ""
    budget_tokens: int = 0
    encoding_format: str = ""
    candidate_set_size: int = 0
    seed: int = 0

    # ── Instance-level
    instance_id: str = ""
    step_t: int = 0
    total_steps: int = 0

    # ── Message (sender → receiver)
    message_raw: str = ""
    message_tokens: int = 0
    message_truncated: bool = False
    budget_enforcement_mode: str = ""
    message_budget_violated: bool = False
    canonical_payload_extracted: bool = False
    canonical_payload_source: str = ""
    canonical_payload_tokens: int = 0

    # ── Receiver I/O
    receiver_input: str = ""          # what receiver actually sees (for reproducibility)
    receiver_input_tokens: int = 0
    receiver_output: str = ""
    receiver_output_tokens: int = 0

    # ── Coordination target prediction
    action_type_pred: str = ""
    artifact_pred: str = ""
    action_type_gold: str = ""
    artifact_gold: str = ""
    hit_action: bool = False
    hit_artifact: bool = False
    hit_joint: bool = False

    # ── Log-prob based metrics (for CSI computation)
    surprisal_bits: float = 0.0       # −log₂ p(z*|X,M)
    surprisal_no_msg: float = 0.0     # −log₂ p(z*|X)  [no-message baseline]
    csi_sample: float = 0.0           # surprisal_no_msg − surprisal_bits
    scoring_mode: str = ""            # "exact" | "fallback_chat" | "proxy" — how CSI was computed

    # ── Receiver logprobs over full candidate set
    candidate_set: list[str] | None = None        # the candidate strings
    candidate_logprobs: dict | None = None         # {label_idx: log_prob}
    scoring_latency_ms: int = 0

    # ── Perturbation tracking (for P3)
    is_perturbed: bool = False
    perturbation_type: str = ""       # "omit_field" | "corrupt_artifact" | ...
    horizon_after_perturb: int = 0
    downstream_inconsistencies: int = 0

    # ── Repair tracking (for P5)
    repair_mode: str = ""             # "sparse_critical" | "random" | "none"
    repair_k: int = 0
    repaired_fields: list[str] = field(default_factory=list)
    escalation_applied: bool = False

    # ── Performance
    latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    timestamp: float = field(default_factory=time.time)

    # ── Task outcome (filled at end of instance)
    task_success: bool = False
    task_score: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove large fields if logging is compact
        if d.get("candidate_logprobs") is None:
            d.pop("candidate_logprobs", None)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ── Instance-Level Summary ────────────────────────────

@dataclass
class InstanceResult:
    """Aggregated result for one benchmark instance."""
    run_id: str
    instance_id: str
    benchmark: str
    agent_type: str
    task_success: bool
    task_score: float
    total_steps: int
    mean_csi: float
    mean_surprisal: float
    mean_message_tokens: int
    total_latency_ms: int
    debt_trajectory: list[float] = field(default_factory=list)
    amplification_trajectory: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
