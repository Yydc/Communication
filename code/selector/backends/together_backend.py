"""Together AI backend for the selector.

Together AI exposes an OpenAI-compatible chat-completions API at
``https://api.together.xyz/v1``, so this backend is a thin wrapper around
the same `openai` SDK with a different base URL and key.

The backend is appropriate for hosted open-weight runs (Qwen, Llama,
DeepSeek, Mixtral, etc.) used in the paper's Qwen rows.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from core.model_interface import BaseModelBackend
from core.types import ModelRequest, ModelResponse


class TogetherBackend(BaseModelBackend):
    """Together AI chat-completions backend (OpenAI-compatible).

    Args:
        model_name: Together model id, e.g. ``Qwen/Qwen3.5-9B-Instruct`` or
            ``meta-llama/Llama-3.3-70B-Instruct-Turbo``.
        api_key_env: environment variable holding the Together API key.
        base_url: defaults to ``https://api.together.xyz/v1``.
        timeout_s: per-request timeout.
        max_retries: retries on transient errors.
    """

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "TOGETHER_API_KEY",
        base_url: str = "https://api.together.xyz/v1",
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__(model_name=model_name)
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is not set; "
                "TogetherBackend cannot be constructed."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai>=1.0 is required for TogetherBackend (Together's "
                "API is OpenAI-compatible). Install with `pip install openai`."
            ) from e
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_s,
            max_retries=max_retries,
        )

    # ---------------------------------------------------------------- generate

    def generate(self, request: ModelRequest) -> ModelResponse:
        kwargs = {
            "model": request.model_name or self.model_name,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.stop:
            kwargs["stop"] = request.stop

        t0 = time.time()
        completion = self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        choice = completion.choices[0]
        text = choice.message.content or ""
        usage = getattr(completion, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0

        return ModelResponse(
            content=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            logprobs=None,
            latency_ms=latency_ms,
            model_name=kwargs["model"],
            raw_response=None,
        )

    # ---------------------------------------------------------------- score

    def score_candidates(self, prompt: str, candidate_labels: list[str]):
        raise NotImplementedError(
            "TogetherBackend.score_candidates is intentionally not implemented; "
            "the controller only requires .generate()."
        )
