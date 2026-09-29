"""Anthropic (Claude) backend for the selector.

Implements `core.model_interface.BaseModelBackend.generate()` using the
official `anthropic` Python SDK. Claude's `messages` API differs from
OpenAI's chat-completions in two relevant ways:

  1. The system prompt is a top-level argument, not a message with
     ``role="system"``.
  2. Tool/function calls are surfaced as a structured ``content`` list
     rather than a string. the controller only uses plain-text completion, so we
     join text blocks and ignore tool blocks.

Tested with `anthropic >= 0.40`.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from core.model_interface import BaseModelBackend
from core.types import ModelRequest, ModelResponse


def _split_system(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    """Pull a leading system message out of an OpenAI-style messages list."""
    system_text: Optional[str] = None
    rest: list[dict] = []
    for m in messages:
        if m.get("role") == "system" and system_text is None:
            system_text = m.get("content", "")
        else:
            rest.append({"role": m["role"], "content": m["content"]})
    # Anthropic requires the conversation to start with a user message.
    if rest and rest[0]["role"] != "user":
        rest = [{"role": "user", "content": ""}] + rest
    return system_text, rest


class AnthropicBackend(BaseModelBackend):
    """Anthropic Claude messages backend.

    Args:
        model_name: a Claude model id, e.g. ``claude-3-5-sonnet-20241022``
            or ``claude-3-7-sonnet-20250219``.
        api_key_env: environment variable holding the Anthropic API key.
        base_url: optional override for the API base URL.
        timeout_s: per-request timeout.
        max_retries: retries on transient errors.
    """

    def __init__(
        self,
        model_name: str,
        api_key_env: str = "ANTHROPIC_API_KEY",
        base_url: Optional[str] = None,
        timeout_s: float = 120.0,
        max_retries: int = 3,
    ):
        super().__init__(model_name=model_name)
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(
                f"Environment variable {api_key_env!r} is not set; "
                "AnthropicBackend cannot be constructed."
            )
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic>=0.40 is required for AnthropicBackend. "
                "Install with `pip install anthropic`."
            ) from e
        kwargs = {"api_key": api_key, "timeout": timeout_s, "max_retries": max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Anthropic(**kwargs)

    # ---------------------------------------------------------------- generate

    def generate(self, request: ModelRequest) -> ModelResponse:
        system_text, msgs = _split_system(request.messages)

        kwargs = {
            "model": request.model_name or self.model_name,
            "messages": msgs,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if system_text:
            kwargs["system"] = system_text
        if request.stop:
            kwargs["stop_sequences"] = request.stop

        t0 = time.time()
        completion = self._client.messages.create(**kwargs)
        latency_ms = int((time.time() - t0) * 1000)

        # `content` is a list of blocks; concatenate text-typed blocks.
        text_parts: list[str] = []
        for block in completion.content:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
        text = "".join(text_parts)

        usage = getattr(completion, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0

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
            "AnthropicBackend.score_candidates is intentionally not implemented; "
            "the controller only requires .generate(). Claude does not expose a clean "
            "per-token logprob surface today."
        )
