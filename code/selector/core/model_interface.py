"""
core/model_interface.py — Unified LLM caller for local and API backends.

Usage:
    model = ModelInterface.from_config(model_cfg)
    resp = model.generate(ModelRequest(...))
    logprobs = model.score_candidates(prompt, candidate_labels)

Backends:
    - SGLangBackend: local SGLang server (OpenAI-compatible)
    - VLLMBackend: local vLLM server (OpenAI-compatible)
    - APIBackend: remote OpenAI-compatible API (Together, etc.)
"""

from __future__ import annotations
import os
import time
import json
import math
import logging
from typing import Optional
from abc import ABC, abstractmethod

from core.types import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class BaseModelBackend(ABC):
    """Abstract backend for LLM inference."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(self, request: ModelRequest) -> ModelResponse:
        ...

    @abstractmethod
    def score_candidates(
        self,
        prompt_messages: list[dict],
        candidate_strings: list[str],
    ) -> dict[int, float]:
        """
        Return {candidate_index: log_prob} for each candidate completion.

        Implementations should compute log p(candidate | prompt) for each candidate.
        The log_prob is in natural log (ln) by default from most APIs.
        """
        ...

    def scoring_mode(self) -> str:
        """Return the scoring mode identifier for logging."""
        return "unknown"


# ── Shared scoring utilities ──────────────────────────────────

def _score_via_completion_api(
    client,
    model_path: str,
    base_prompt: str,
    candidate_strings: list[str],
    batch_size: int = 16,
) -> dict[int, float]:
    """
    Score candidates using the /v1/completions endpoint with echo + logprobs.

    For each candidate, we send prompt + candidate with max_tokens=0 and echo=True.
    The returned logprobs over the candidate tokens give us log p(candidate | prompt).

    Supports batching: sends `batch_size` candidates per API call where supported.
    Falls back to serial requests if batching fails.
    """
    scores = {}

    # Try batch scoring first (SGLang/vLLM support batched prompts)
    try:
        return _batch_score(client, model_path, base_prompt, candidate_strings, batch_size)
    except Exception as e:
        logger.debug(f"Batch scoring failed, falling back to serial: {e}")

    # Serial fallback
    for idx, cand in enumerate(candidate_strings):
        try:
            resp = client.completions.create(
                model=model_path,
                prompt=base_prompt + cand,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )
            choice = resp.choices[0]
            if choice.logprobs and choice.logprobs.token_logprobs:
                all_lp = choice.logprobs.token_logprobs
                all_tokens = choice.logprobs.tokens or []
                # Count how many tokens belong to the candidate
                # Use the tokens list to find where the candidate starts
                n_cand_tokens = _count_candidate_tokens(all_tokens, cand)
                cand_lps = [lp for lp in all_lp[-n_cand_tokens:] if lp is not None]
                scores[idx] = sum(cand_lps) if cand_lps else -100.0
            else:
                scores[idx] = -100.0
        except Exception as e:
            logger.warning(f"Candidate scoring failed for idx={idx}: {e}")
            scores[idx] = -100.0

    return scores


def _batch_score(
    client,
    model_path: str,
    base_prompt: str,
    candidate_strings: list[str],
    batch_size: int,
) -> dict[int, float]:
    """
    Batch-score candidates by sending multiple prompts in one API call.
    SGLang and vLLM both support list-of-prompts in the completions API.
    """
    scores = {}

    for batch_start in range(0, len(candidate_strings), batch_size):
        batch = candidate_strings[batch_start:batch_start + batch_size]
        prompts = [base_prompt + cand for cand in batch]

        resp = client.completions.create(
            model=model_path,
            prompt=prompts,
            max_tokens=0,
            echo=True,
            logprobs=1,
        )

        for i, choice in enumerate(resp.choices):
            idx = batch_start + i
            if choice.logprobs and choice.logprobs.token_logprobs:
                all_lp = choice.logprobs.token_logprobs
                all_tokens = choice.logprobs.tokens or []
                cand = batch[i]
                n_cand_tokens = _count_candidate_tokens(all_tokens, cand)
                cand_lps = [lp for lp in all_lp[-n_cand_tokens:] if lp is not None]
                scores[idx] = sum(cand_lps) if cand_lps else -100.0
            else:
                scores[idx] = -100.0

    return scores


def _count_candidate_tokens(all_tokens: list[str], candidate: str) -> int:
    """
    Determine how many trailing tokens in all_tokens correspond to the candidate string.
    Reconstructs from the end to find the boundary.
    """
    if not all_tokens:
        return 1

    # Walk backwards through tokens, accumulating text
    accumulated = ""
    count = 0
    for token in reversed(all_tokens):
        accumulated = token + accumulated
        count += 1
        # Stop when we've covered at least the candidate length
        if len(accumulated.strip()) >= len(candidate.strip()):
            break

    return max(1, count)


def _messages_to_prompt(messages: list[dict]) -> str:
    """Convert chat messages to a flat prompt string for completion API."""
    parts = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            parts.append(f"<|system|>\n{content}\n")
        elif role == "user":
            parts.append(f"<|user|>\n{content}\n")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}\n")
    parts.append("<|assistant|>\n")
    return "".join(parts)


def _score_via_chat_api(
    client,
    model_name: str,
    prompt_messages: list[dict],
    candidate_strings: list[str],
    top_k: int = 20,
) -> dict[int, float]:
    """
    Score candidates using the chat completions API with logprobs.

    Strategy: ask the model to output one of the candidates, then use
    the logprobs of the first token(s) to estimate p(candidate | prompt).

    This is a FALLBACK when the completion API is not available (e.g., pure API backends).
    It's less accurate than the completion-based method but works with any OpenAI-compatible API.
    """
    # Build a prompt that lists candidates and asks the model to pick
    candidate_list = "\n".join(f"{i}: {c}" for i, c in enumerate(candidate_strings[:top_k]))
    pick_message = (
        f"Given the candidates below, output ONLY the number of the best candidate.\n\n"
        f"Candidates:\n{candidate_list}\n\nYour answer (number only):"
    )

    messages = prompt_messages + [{"role": "user", "content": pick_message}]

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=5,
            temperature=0.0,
            logprobs=True,
            top_logprobs=top_k,
        )

        choice = resp.choices[0]
        scores = {i: -100.0 for i in range(len(candidate_strings))}

        # Parse logprobs from the first token
        if choice.logprobs and hasattr(choice.logprobs, "content") and choice.logprobs.content:
            first_token_lps = choice.logprobs.content[0]
            if first_token_lps.top_logprobs:
                for tlp in first_token_lps.top_logprobs:
                    # Try to match token to candidate index
                    try:
                        idx = int(tlp.token.strip())
                        if 0 <= idx < len(candidate_strings):
                            scores[idx] = tlp.logprob
                    except (ValueError, IndexError):
                        pass

        return scores

    except Exception as e:
        logger.warning(f"Chat-based candidate scoring failed: {e}")
        return {i: -100.0 for i in range(len(candidate_strings))}


# ══════════════════════════════════════════════════════════════
# SGLang Backend
# ══════════════════════════════════════════════════════════════

class SGLangBackend(BaseModelBackend):
    """
    Local inference via SGLang Runtime.
    Expects SGLang server running at http://{host}:{port}/v1
    """

    def __init__(self, model_name: str, endpoint: str, port: int = 30000, **kwargs):
        super().__init__(model_name)
        host = kwargs.get("host") or os.getenv("MALLM_MODEL_HOST", "localhost")
        self.base_url = f"http://{host}:{port}/v1"
        self.model_path = endpoint
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    base_url=self.base_url,
                    api_key="EMPTY",
                )
            except ImportError:
                raise ImportError("pip install openai  # needed for SGLang OpenAI-compatible API")
        return self._client

    def generate(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        t0 = time.perf_counter()

        # Reasoning and GPT-5+ models require max_completion_tokens, no temperature
        is_new_openai = any(self.model_path.startswith(p) for p in ["gpt-5", "o3", "o4", "o1"])
        token_key = "max_completion_tokens" if is_new_openai else "max_tokens"

        kwargs = dict(
            model=self.model_path,
            messages=request.messages,
        )
        if not is_new_openai:
            kwargs["temperature"] = request.temperature
            kwargs["top_p"] = request.top_p
        kwargs[token_key] = request.max_tokens
        if request.stop:
            kwargs["stop"] = request.stop
        if request.logprobs and not is_new_openai:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 20

        resp = client.chat.completions.create(**kwargs)
        latency = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        logprob_data = None
        if request.logprobs and choice.logprobs:
            logprob_data = self._parse_logprobs(choice.logprobs)

        return ModelResponse(
            content=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            logprobs=logprob_data,
            latency_ms=latency,
            model_name=self.model_name,
        )

    def score_candidates(
        self,
        prompt_messages: list[dict],
        candidate_strings: list[str],
    ) -> dict[int, float]:
        """
        Exact candidate scoring via completion API with echo + logprobs.
        Uses batched requests for efficiency.
        """
        client = self._get_client()
        base_prompt = _messages_to_prompt(prompt_messages)
        return _score_via_completion_api(
            client, self.model_path, base_prompt, candidate_strings, batch_size=32
        )

    def scoring_mode(self) -> str:
        return "exact"

    @staticmethod
    def _parse_logprobs(logprobs_obj) -> dict:
        """Parse OpenAI-format logprobs into a simple dict."""
        result = {}
        if hasattr(logprobs_obj, "content") and logprobs_obj.content:
            for token_lp in logprobs_obj.content:
                result[token_lp.token] = token_lp.logprob
                if token_lp.top_logprobs:
                    for tlp in token_lp.top_logprobs:
                        result[tlp.token] = tlp.logprob
        return result


# ══════════════════════════════════════════════════════════════
# vLLM Backend
# ══════════════════════════════════════════════════════════════

class VLLMBackend(BaseModelBackend):
    """
    Local inference via vLLM.
    Expects vLLM server running at http://{host}:{port}/v1
    """

    def __init__(self, model_name: str, endpoint: str, port: int = 8000, **kwargs):
        super().__init__(model_name)
        host = kwargs.get("host") or os.getenv("MALLM_MODEL_HOST", "localhost")
        self.base_url = f"http://{host}:{port}/v1"
        self.model_path = endpoint
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key="EMPTY")
        return self._client

    def generate(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        t0 = time.perf_counter()

        kwargs = dict(
            model=self.model_path,
            messages=request.messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )
        if request.logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = 20

        resp = client.chat.completions.create(**kwargs)
        latency = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        logprob_data = None
        if request.logprobs and choice.logprobs:
            logprob_data = SGLangBackend._parse_logprobs(choice.logprobs)

        return ModelResponse(
            content=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            logprobs=logprob_data,
            latency_ms=latency,
            model_name=self.model_name,
        )

    def score_candidates(
        self,
        prompt_messages: list[dict],
        candidate_strings: list[str],
    ) -> dict[int, float]:
        """
        Exact candidate scoring via vLLM's completion API.
        Same interface as SGLang (both are OpenAI-compatible).
        """
        client = self._get_client()
        base_prompt = _messages_to_prompt(prompt_messages)
        return _score_via_completion_api(
            client, self.model_path, base_prompt, candidate_strings, batch_size=32
        )

    def scoring_mode(self) -> str:
        return "exact"


# ══════════════════════════════════════════════════════════════
# API Backend (Together AI, OpenAI-compatible remote)
# ══════════════════════════════════════════════════════════════

class APIBackend(BaseModelBackend):
    """
    Remote API backend (Together AI, OpenAI-compatible, etc.)
    """

    def __init__(self, model_name: str, endpoint: str, api_key_env: str,
                 supports_logprobs: bool = True, max_logprobs: int = 20, **kwargs):
        super().__init__(model_name)
        self.endpoint = endpoint
        self.api_key = os.environ.get(api_key_env, "")
        if not self.api_key:
            logger.warning(f"API key env var '{api_key_env}' not set")
        self._client = None
        # Check if this API supports the completion endpoint (for exact scoring)
        self._supports_completions = None
        # Whether this model's API supports logprobs (some providers cap at <20)
        self._supports_logprobs = supports_logprobs
        self._max_logprobs = max_logprobs

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.endpoint, api_key=self.api_key)
        return self._client

    def generate(self, request: ModelRequest) -> ModelResponse:
        client = self._get_client()
        t0 = time.perf_counter()

        # GPT-5+ / o3 / o4 models require max_completion_tokens instead of max_tokens
        # Reasoning and GPT-5+ models require max_completion_tokens, no temperature
        is_new_openai = any(self.model_name.startswith(p) for p in ["gpt-5", "o3", "o4", "o1"])
        token_key = "max_completion_tokens" if is_new_openai else "max_tokens"

        kwargs = dict(
            model=self.model_name,
            messages=request.messages,
        )
        if not is_new_openai:
            kwargs["temperature"] = request.temperature
        kwargs[token_key] = request.max_tokens

        if request.logprobs and self._supports_logprobs and not is_new_openai:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self._max_logprobs

        resp = client.chat.completions.create(**kwargs)
        latency = int((time.perf_counter() - t0) * 1000)

        choice = resp.choices[0]
        logprob_data = None
        if request.logprobs and choice.logprobs:
            logprob_data = SGLangBackend._parse_logprobs(choice.logprobs)

        return ModelResponse(
            content=choice.message.content or "",
            input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            logprobs=logprob_data,
            latency_ms=latency,
            model_name=self.model_name,
        )

    def score_candidates(
        self,
        prompt_messages: list[dict],
        candidate_strings: list[str],
    ) -> dict[int, float]:
        """
        Score candidates via API.

        Strategy:
        1. Try completion API (exact) — works with Together AI and some OpenAI-compatible APIs
        2. Fall back to chat API with logprobs (approximate)
        """
        client = self._get_client()

        # Try exact scoring via completion API first
        if self._supports_completions is not False:
            try:
                base_prompt = _messages_to_prompt(prompt_messages)
                scores = _score_via_completion_api(
                    client, self.model_name, base_prompt, candidate_strings, batch_size=8
                )
                self._supports_completions = True
                return scores
            except Exception as e:
                logger.info(f"Completion API not available for {self.model_name}: {e}")
                self._supports_completions = False

        # Fallback: chat API with logprobs
        return _score_via_chat_api(
            client, self.model_name, prompt_messages, candidate_strings
        )

    def scoring_mode(self) -> str:
        if self._supports_completions:
            return "exact"
        return "fallback_chat"


# ══════════════════════════════════════════════════════════════
# Transformers Backend
# ══════════════════════════════════════════════════════════════

class TransformersBackend(BaseModelBackend):
    """Direct local inference via Hugging Face Transformers."""

    def __init__(self, model_name: str, endpoint: str, **kwargs):
        super().__init__(model_name)
        self.model_path = endpoint
        self.device = kwargs.get("device", "cuda:0")
        self.max_ctx_tokens = int(kwargs.get("max_ctx_tokens", 32768))
        self.dtype_name = kwargs.get("dtype", "bfloat16")
        self.attn_implementation = kwargs.get("attn_implementation", "sdpa")
        self.scoring_batch_size = int(kwargs.get("scoring_batch_size", 8))
        self.trust_remote_code = bool(kwargs.get("trust_remote_code", True))
        self._tokenizer = None
        self._model = None
        self._torch = None

    def _get_torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    def _resolve_dtype(self):
        torch = self._get_torch()
        if self.dtype_name == "float16":
            return torch.float16
        if self.dtype_name == "float32":
            return torch.float32
        return torch.bfloat16

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch = self._get_torch()
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs = dict(
            torch_dtype=self._resolve_dtype(),
            trust_remote_code=self.trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if self.attn_implementation:
            load_kwargs["attn_implementation"] = self.attn_implementation

        if self.device.startswith("cuda"):
            load_kwargs["device_map"] = {"": self.device}

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            **load_kwargs,
        )

        model_device = getattr(self._model, "device", None)
        if model_device is None or str(model_device) == "cpu":
            self._model.to(self.device)

        self._model.eval()
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            torch.cuda.empty_cache()

    def _chat_prompt_text(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
        self._load()
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _encode_prompt(self, messages: list[dict]):
        self._load()
        prompt_text = self._chat_prompt_text(messages, add_generation_prompt=True)
        encoded = self._tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        return prompt_text, encoded

    def _truncate_model_inputs(self, input_ids, attention_mask, labels=None):
        max_len = self.max_ctx_tokens
        if input_ids.shape[1] <= max_len:
            return input_ids, attention_mask, labels
        input_ids = input_ids[:, -max_len:]
        attention_mask = attention_mask[:, -max_len:]
        if labels is not None:
            labels = labels[:, -max_len:]
        return input_ids, attention_mask, labels

    def _move_to_device(self, tensor_dict: dict):
        self._load()
        return {k: v.to(self.device) for k, v in tensor_dict.items()}

    def generate(self, request: ModelRequest) -> ModelResponse:
        self._load()
        torch = self._get_torch()
        t0 = time.perf_counter()

        _, encoded = self._encode_prompt(request.messages)
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        input_ids, attention_mask, _ = self._truncate_model_inputs(input_ids, attention_mask)

        model_inputs = self._move_to_device({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        })

        do_sample = request.temperature > 0.0
        generate_kwargs = dict(
            max_new_tokens=request.max_tokens,
            do_sample=do_sample,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=request.logprobs,
        )
        if do_sample:
            generate_kwargs["temperature"] = request.temperature
            generate_kwargs["top_p"] = request.top_p
        if request.stop:
            generate_kwargs["stop_strings"] = request.stop
            generate_kwargs["tokenizer"] = self._tokenizer

        with torch.inference_mode():
            generated = self._model.generate(**model_inputs, **generate_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        generated_ids = generated.sequences[0, prompt_len:]
        content = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        logprob_data = None
        if request.logprobs and generated_ids.numel() > 0 and getattr(generated, "scores", None):
            transition_scores = self._model.compute_transition_scores(
                generated.sequences,
                generated.scores,
                normalize_logits=True,
            )
            token_scores = transition_scores[0, -generated_ids.shape[0]:].tolist()
            logprob_data = {}
            for token_id, token_logprob in zip(generated_ids.tolist(), token_scores):
                token_text = self._tokenizer.decode([token_id], skip_special_tokens=False)
                logprob_data[token_text] = float(token_logprob)

        latency = int((time.perf_counter() - t0) * 1000)
        return ModelResponse(
            content=content,
            input_tokens=int(model_inputs["input_ids"].shape[1]),
            output_tokens=int(generated_ids.shape[0]),
            logprobs=logprob_data,
            latency_ms=latency,
            model_name=self.model_name,
        )

    def score_candidates(
        self,
        prompt_messages: list[dict],
        candidate_strings: list[str],
    ) -> dict[int, float]:
        self._load()
        torch = self._get_torch()
        prompt_text = self._chat_prompt_text(prompt_messages, add_generation_prompt=True)
        prompt_ids = self._tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        pad_token_id = self._tokenizer.pad_token_id
        results = {}

        for batch_start in range(0, len(candidate_strings), self.scoring_batch_size):
            batch = candidate_strings[batch_start:batch_start + self.scoring_batch_size]
            batch_input_ids = []
            batch_attention = []
            batch_labels = []
            max_len = 0

            encoded_candidates = [
                self._tokenizer(cand, add_special_tokens=False)["input_ids"]
                for cand in batch
            ]

            for cand_ids in encoded_candidates:
                full_ids = prompt_ids + cand_ids
                labels = ([-100] * len(prompt_ids)) + cand_ids
                full_len = len(full_ids)

                if full_len > self.max_ctx_tokens:
                    full_ids = full_ids[-self.max_ctx_tokens:]
                    labels = labels[-self.max_ctx_tokens:]
                    full_len = len(full_ids)

                max_len = max(max_len, full_len)
                batch_input_ids.append(full_ids)
                batch_labels.append(labels)

            for idx in range(len(batch_input_ids)):
                seq_len = len(batch_input_ids[idx])
                pad_len = max_len - seq_len
                batch_input_ids[idx] = batch_input_ids[idx] + ([pad_token_id] * pad_len)
                batch_labels[idx] = batch_labels[idx] + ([-100] * pad_len)
                batch_attention.append(([1] * seq_len) + ([0] * pad_len))

            input_ids = torch.tensor(batch_input_ids, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(batch_attention, dtype=torch.long, device=self.device)
            labels = torch.tensor(batch_labels, dtype=torch.long, device=self.device)

            with torch.inference_mode():
                logits = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits
                log_probs = logits.log_softmax(dim=-1)

            shift_log_probs = log_probs[:, :-1, :]
            shift_labels = labels[:, 1:]
            shift_mask = shift_labels.ne(-100)

            safe_labels = shift_labels.masked_fill(~shift_mask, 0)
            token_log_probs = shift_log_probs.gather(
                dim=-1,
                index=safe_labels.unsqueeze(-1),
            ).squeeze(-1)
            token_log_probs = token_log_probs * shift_mask
            seq_scores = token_log_probs.sum(dim=-1).tolist()

            for local_idx, score in enumerate(seq_scores):
                results[batch_start + local_idx] = float(score)

        return results

    def scoring_mode(self) -> str:
        return "exact"


# ── Factory ───────────────────────────────────────────

class ModelInterface:
    """
    Factory that creates the right backend from config.

    Usage:
        model = ModelInterface.from_config(cfg)
        resp = model.generate(ModelRequest(...))
    """

    @staticmethod
    def from_config(model_cfg: dict, engine: str = "sglang") -> BaseModelBackend:
        """
        model_cfg: one entry from configs/default.yaml → models.ladder[i]
        engine: "sglang" | "vllm" | "transformers" (for local models)
        """
        backend = model_cfg.get("backend", "local")
        name = model_cfg["name"]
        endpoint = model_cfg.get("path_or_endpoint", "")

        if backend == "local":
            if engine == "sglang":
                return SGLangBackend(
                    model_name=name,
                    endpoint=endpoint,
                    port=model_cfg.get("port", 30000),
                )
            elif engine == "vllm":
                return VLLMBackend(
                    model_name=name,
                    endpoint=endpoint,
                    port=model_cfg.get("port", 8000),
                )
            elif engine == "transformers":
                return TransformersBackend(
                    model_name=name,
                    endpoint=endpoint,
                    device=model_cfg.get("device", "cuda:0"),
                    max_ctx_tokens=model_cfg.get("max_ctx_tokens", 32768),
                    dtype=model_cfg.get("dtype", "bfloat16"),
                    attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
                    scoring_batch_size=model_cfg.get("scoring_batch_size", 8),
                    trust_remote_code=model_cfg.get("trust_remote_code", True),
                )
            else:
                raise ValueError(f"Unknown local engine: {engine}")

        elif backend == "api":
            return APIBackend(
                model_name=name,
                endpoint=endpoint,
                api_key_env=model_cfg.get("api_key_env", "API_KEY"),
                supports_logprobs=model_cfg.get("supports_logprobs", True),
                max_logprobs=model_cfg.get("max_logprobs", 20),
            )

        else:
            raise ValueError(f"Unknown backend: {backend}")

    @staticmethod
    def from_name(
        model_name: str,
        all_models: list[dict],
        engine: str = "sglang",
    ) -> BaseModelBackend:
        """Look up a model by name from the full model list."""
        for cfg in all_models:
            if cfg["name"] == model_name:
                return ModelInterface.from_config(cfg, engine)
        raise ValueError(f"Model '{model_name}' not found in config")
