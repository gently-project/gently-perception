"""
Core types for the perception harness.

Design invariants enforced here:
- FrameInput is frozen and carries NO ground-truth field — the GT leak surface
  from the old run.py:126-131 cannot exist because there's nowhere to put GT.
- History is an immutable tuple of predicted Observations.
- Prediction.stage is a Stage enum value, not a free string — the model returns
  it via a forced classify_stage tool call, so an invalid stage is impossible.
"""
from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class Stage(str, Enum):
    """C. elegans embryo developmental stages, in order."""

    EARLY = "early"
    BEAN = "bean"
    COMMA = "comma"
    ONE_HALF_FOLD = "1.5fold"
    TWO_FOLD = "2fold"
    PRETZEL = "pretzel"
    HATCHING = "hatching"
    HATCHED = "hatched"

    @property
    def ordinal(self) -> int:
        return STAGE_ORDER.index(self)

    def distance(self, other: "Stage") -> int:
        return abs(self.ordinal - other.ordinal)


STAGE_ORDER: tuple[Stage, ...] = tuple(Stage)

# Typical durations in minutes — from gently_perception/organism.py CELEGANS.
STAGE_DURATIONS_MIN: dict[Stage, int] = {
    Stage.EARLY: 350,
    Stage.BEAN: 30,
    Stage.COMMA: 30,
    Stage.ONE_HALF_FOLD: 60,
    Stage.TWO_FOLD: 90,
    Stage.PRETZEL: 300,
    Stage.HATCHING: 20,
    Stage.HATCHED: 0,
}


@dataclass(frozen=True, slots=True)
class Observation:
    """A single past prediction, used to build temporal history."""

    timepoint: int
    stage: Stage
    timestamp: float


@dataclass(frozen=True, slots=True)
class FrameInput:
    """Everything a solver sees for one frame.

    Frozen and immutable: solvers cannot mutate history or smuggle state across
    frames. Carries no ground-truth field by construction.
    """

    embryo_id: str
    timepoint: int
    image_b64: str
    """Default 3-view max-projection JPEG, base64-encoded."""
    volume_ref: Path
    """Path to the raw 3D TIFF — tools load on demand."""
    references: Mapping[Stage, tuple[str, ...]]
    """Reference images per stage, base64-encoded."""
    history: tuple[Observation, ...]
    """Predicted observations for this embryo so far. Never contains GT."""
    history_text: str
    """Pre-rendered last-N history string so solvers don't loop over history."""
    last_stage: Stage | None
    """Convenience: history[-1].stage, or None if no history."""
    prev_volume_refs: tuple[Path, ...] = ()
    """Paths to the last K volumes, for the prev_frame tool."""


# --- Tool results -----------------------------------------------------------


@dataclass(frozen=True)
class ImageResult:
    kind: Literal["image"] = field(default="image", init=False)
    b64: str = ""
    caption: str = ""


@dataclass(frozen=True)
class NumericResult:
    kind: Literal["numeric"] = field(default="numeric", init=False)
    value: float = 0.0
    unit: str = ""
    note: str = ""


@dataclass(frozen=True)
class ErrorResult:
    kind: Literal["error"] = field(default="error", init=False)
    message: str = ""


ToolResult = ImageResult | NumericResult | ErrorResult


# --- Prediction & trajectory ------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    stage: Stage
    reasoning: str
    raw: dict[str, Any]
    """Verbatim classify_stage tool input block."""


@dataclass
class Step:
    """One step of the agent loop — model turn or tool dispatch."""

    kind: Literal["model", "tool"]
    tool_name: str | None = None
    input: dict[str, Any] | None = None
    output_kind: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def model(cls, resp: Any) -> "Step":
        usage = getattr(resp, "usage", None)
        return cls(
            kind="model",
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    @classmethod
    def tool(cls, name: str, params: dict[str, Any], result: ToolResult) -> "Step":
        return cls(kind="tool", tool_name=name, input=params, output_kind=result.kind)


@dataclass
class Trajectory:
    """Per-frame agent transcript.

    Modeled on inspect-ai's EvalSample and agents_core's ThreadPiece: full
    message list for replay plus a structured step list for analysis.
    """

    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    budget_exhausted: bool = False

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool")

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.steps)


class ModelOutputError(RuntimeError):
    """Raised when the model fails to produce a tool_use block.

    Never silently coerced to a fake prediction — the frame is recorded as an
    error event and excluded from the accuracy denominator.
    """

    def __init__(self, reason: str, response: Any = None) -> None:
        super().__init__(reason)
        self.response = response


def now() -> float:
    return time.time()
