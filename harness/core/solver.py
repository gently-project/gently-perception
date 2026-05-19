"""
Solver — the declarative variant config.

A solver is *data*, not behavior: a system prompt (or a pure function of
FrameInput → str for stage-conditioned prompts like hybrid), a tuple of tool
names, and a step budget. The agent loop, history, rendering, verification, and
scoring are all harness-owned and unreachable from a Solver.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from harness.core.model import DEFAULT_MODEL
from harness.core.types import FrameInput

SystemPrompt = str | Callable[[FrameInput], str]


@dataclass(frozen=True)
class Solver:
    name: str
    system: SystemPrompt
    tools: tuple[str, ...] = ()
    max_steps: int = 1
    """Tool-call budget. 1 with tools=() ⇒ one-shot forced classify (parity mode)."""
    model: str = DEFAULT_MODEL
    thinking: str | None = None
    """None | "adaptive". None for parity solvers; "adaptive" for agentic."""
    effort: str | None = None
    """None | "low" | "medium" | "high" | "max". None for parity."""

    def system_for(self, frame: FrameInput) -> str:
        return self.system(frame) if callable(self.system) else self.system

    @property
    def is_one_shot(self) -> bool:
        return not self.tools and self.max_steps <= 1


def solver(**kw) -> Solver:
    """Convenience constructor for solvers/ modules."""
    return Solver(**kw)
