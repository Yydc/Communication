"""Concrete LLM backends for the selector controller.

Three reference implementations are provided:

* OpenAIBackend     — chat.completions over the official OpenAI API
* TogetherBackend   — chat.completions over Together AI's OpenAI-compatible endpoint
* AnthropicBackend  — messages over the official Anthropic API

All three implement core.model_interface.BaseModelBackend.generate(). Only
generate() is needed by SelectorController; score_candidates() raises
NotImplementedError, since none of the three providers exposes a uniform
constrained-token-logprob interface today.

Each backend reads its API key from an environment variable. No credentials
are accepted as arguments or read from chat history.
"""

from .openai_backend import OpenAIBackend
from .together_backend import TogetherBackend
from .anthropic_backend import AnthropicBackend

__all__ = ["OpenAIBackend", "TogetherBackend", "AnthropicBackend"]
