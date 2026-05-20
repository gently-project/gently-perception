"""
Solver — the declarative variant config.

A solver is *data*, not behavior: a system prompt (or a pure function of
FrameInput → str for stage-conditioned prompts like hybrid), a tuple of tool
names, and a step budget. The agent loop, history, rendering, verification, and
scoring are all harness-owned and unreachable from a Solver.

`user_blocks` lets a solver shape the dynamic user-turn content (header text,
history rendering, anchoring, image position, trailing CoT prompt). This is
prompt strategy — it belongs to the solver. The harness still owns the cached
reference-image prefix and forbids the solver from touching history or GT.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from harness.core.model import DEFAULT_MODEL
from harness.core.types import FrameInput

SystemPrompt = str | Callable[[FrameInput], str]
UserBlocks = Callable[[FrameInput], list[dict[str, Any]]]


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
    user_blocks: UserBlocks | None = None
    """Build the dynamic user-turn content (after the cached refs). If None,
    the harness's default agentic block is used."""

    def system_for(self, frame: FrameInput) -> str:
        return self.system(frame) if callable(self.system) else self.system

    @property
    def is_one_shot(self) -> bool:
        return not self.tools and self.max_steps <= 1


def solver(**kw) -> Solver:
    """Convenience constructor for solvers/ modules."""
    return Solver(**kw)
