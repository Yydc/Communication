"""Selector inner agents.

Currently exposes:
  - MASequentialAgent: standard sender → receiver pipeline that the selector wraps.

The selector dispatches every non-SKIP action through MASequentialAgent.step() to
keep the inner protocol identical to the experiment runs reported in the
paper.
"""

from .ma_sequential import MASequentialAgent

__all__ = ["MASequentialAgent"]
