"""OpenAI chat-completions backend for the selector.

Implements `core.model_interface.BaseModelBackend.generate()` using the
official `openai` Python SDK. The constructor reads the API key from an
environment variable (default `OPENAI_API_KEY`) so no secret is ever
embedded in source or arguments.

Tested with `openai >= 1.0`. Works with reasoning models (`o3`, `o4-mini`,
`gpt-5.4`*) by switching the token-budget keyword to `max_completion_tokens`
and dropping `temperature` when the model rejects it. Supported message
roles: `system`, `user`, `assistant`.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from core.model_interface import BaseModelBackend
from core.types import ModelRequest, ModelResponse


_REASONING_MODEL_PREFIXES = ("o3", "o4-mini", "gpt-5.4")


def _is_reasoning_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _REASONING_MODEL_PREFIXES)


class OpenAIBackend(BaseModelBackend):
    """OpenAI chat-completions backend.

    Args:
        model_name: a dated alias such as ``gpt-4o-mini-2024-07-18`` or a
            general alias (``gpt-4o-mini``). Dated aliases are preferred for
            reproducibility.
        api_key_env: name of the environment variable holding the API key.
        base_url: optional override for the API base URL (default
            ``https://api.openai.com/v1``).
        timeout_s: per-request timeout. Reasoning models can take >60 s.
        max_retries: retries on transient errors (5xx / timeouts).

    Example:
        >>> backend = OpenAIBackend("gpt-4o-mini-2024-07-18")
        >>> resp = backend.generate(ModelRequest(
        ...     messages=[{"role": "user", "content": "hi"}],
        ...     model_name="gpt-4o-mini-2024-07-18",
        ...     max_tokens=64,
        ... ))
        >>> resp.content
    """

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__(model_name=model_name)
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is not set; "
                "OpenAIBackend cannot be constructed."
            )
        # Lazy import so the package is optional unless this backend is used.
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai>=1.0 is required for OpenAIBackend. "
                "Install with `pip install openai`."
            ) from e
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
            timeout=timeout_s,
            max_retries=max_retries,
        )

    # ---------------------------------------------------------------- generate

    def generate(self, request: ModelRequest) -> ModelResponse:
        kwargs = {
            "model": request.model_name or self.model_name,
            "messages": request.messages,
        }
        if request.stop:
            kwargs["stop"] = request.stop
        # Reasoning models reject temperature/top_p and use a different
        # token-budget keyword.
        if _is_reasoning_model(kwargs["model"]):
            kwargs["max_completion_tokens"] = request.max_tokens
        else:
            kwargs["max_tokens"] = request.max_tokens
            kwargs["temperature"] = request.temperature
            kwargs["top_p"] = request.top_p
            if request.logprobs:
                # Leading-token log-prob for the selector's gain probes.
                kwargs["logprobs"] = True

        t0 = time.time()
        completion = self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

        logprobs = None
        content_logprobs = getattr(getattr(choice, "logprobs", None), "content", None)
        if content_logprobs:
            leading = content_logprobs[0]
            logprobs = {leading.token: float(leading.logprob)}

        return ModelResponse(
            content=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            logprobs=logprobs,
            latency_ms=latency_ms,
            model_name=kwargs["model"],
            raw_response=None,
        )

    # ---------------------------------------------------------------- score

    def score_candidates(self, prompt: str, candidate_labels: list[str]):
        """Not used by the controller. OpenAI's logprob surface does not give a clean
        per-candidate score; returning a uniform shim would be misleading,
        so we raise instead."""
        raise NotImplementedError(
            "OpenAIBackend.score_candidates is intentionally not implemented; "
            "the controller only requires .generate()."
        )
