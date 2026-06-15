"""
Multi-task perception types — DESIGN SKETCH, not yet wired.

Extends harness/core/types.py for:
  Q1: a reusable signal-onset abstraction (SignalSpec)
  Q2: routing across multiple perception tasks per frame, with an action output

Everything that already exists in types.py is re-exported unchanged. New types
are grouped by the layer they belong to (observe / task / route / act).

Design invariants carried over:
- All cross-frame state is frozen + immutable (tuples, frozen dataclasses).
- No type here has a ground-truth field. Tasks/routers/policies cannot see GT.
- Model parse failures raise; they are never coerced to a default observation.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable  # noqa: F401

# --- unchanged, re-exported -------------------------------------------------
from harness.core.types import (  # noqa: F401
    STAGE_DURATIONS_MIN,
    STAGE_ORDER,
    ErrorResult,
    ImageResult,
    ModelOutputError,
    NumericResult,
    Stage,
    Step,
    ToolResult,
    Trajectory,
    now,
)


# ==========================================================================
# OBSERVE — what a task emits for one frame
# ==========================================================================

@dataclass(frozen=True, slots=True)
class ObservationBase:
    """One task's read of one frame.

    Replaces the old stage-only ``Observation``. Each task defines its own
    subclass with a typed ``value`` payload; the base carries only what every
    task shares.
    """

    task: str
    timepoint: int
    timestamp: float
    reasoning: str
    raw: Mapping[str, Any]
    """Verbatim model output block (for replay/debugging)."""


@dataclass(frozen=True, slots=True)
class StageObservation(ObservationBase):
    value: Stage


class Intensity(str, Enum):
    """Ordinal signal levels for onset tasks. NONE is always 'absent'."""

    NONE = "none"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"
    SATURATING = "saturating"

    @property
    def ordinal(self) -> int:
        return list(Intensity).index(self)

    def __ge__(self, other: "Intensity") -> bool:  # type: ignore[override]
        return self.ordinal >= other.ordinal


@dataclass(frozen=True, slots=True)
class IntensityObservation(ObservationBase):
    value: Intensity
    onset: bool
    """True on the first frame where value >= the task's threshold."""


Observation = StageObservation | IntensityObservation
ObsT = TypeVar("ObsT", bound=ObservationBase)


# ==========================================================================
# RENDER — per-task volume → image
# ==========================================================================

@dataclass(frozen=True)
class RenderSpec:
    """How a task wants the volume presented.

    Named specs are registered in ``harness/render/`` and resolved by the loop.
    Two tasks with the same spec share one render per frame (cached by name).
    """

    name: str
    """Registry key — e.g. "three_view_mip", "dual_view_calib"."""
    params: Mapping[str, Any] = field(default_factory=dict)


# ==========================================================================
# ACT — perception → microscope
# ==========================================================================

@dataclass(frozen=True)
class SetCadence:
    kind: Literal["set_cadence"] = field(default="set_cadence", init=False)
    interval_s: float
    reason: str = ""


@dataclass(frozen=True)
class SetExposure:
    kind: Literal["set_exposure"] = field(default="set_exposure", init=False)
    ms: float
    reason: str = ""


@dataclass(frozen=True)
class SetROI:
    kind: Literal["set_roi"] = field(default="set_roi", init=False)
    bbox: tuple[int, int, int, int]
    reason: str = ""


@dataclass(frozen=True)
class Halt:
    kind: Literal["halt"] = field(default="halt", init=False)
    reason: str = ""


Action = SetCadence | SetExposure | SetROI | Halt


# ==========================================================================
# TASK — one perception objective
# ==========================================================================

@dataclass(frozen=True, slots=True)
class TaskInput:
    """Everything a task's ``run`` sees for one frame.

    Like ``FrameInput`` but task-scoped: the image is rendered per the task's
    own ``RenderSpec``, and history is this task's own past observations only.
    Frozen — tasks cannot mutate or smuggle state across frames.
    """

    embryo_id: str
    timepoint: int
    timestamp: float
    image_b64: str
    """Rendered per the task's RenderSpec."""
    volume_ref: Path
    own_history: tuple[ObservationBase, ...]
    """This task's prior observations for this embryo. Never contains GT."""
    state: "FrameState"
    """Read-only snapshot of all-task state (e.g. for arming on stage)."""
    description: str | None = None
    """Shared-perceiver free-form prose, when the router provides one."""


TaskRun = Callable[[TaskInput], Awaitable[tuple[ObsT, Trajectory]]]


@dataclass(frozen=True)
class ArmingPolicy:
    """When a task is allowed to run. Biased open — see design notes.

    The loop evaluates this; tasks never gate themselves.
    """

    predicate: Callable[["FrameState"], bool]
    latch: bool = True
    """Once armed for an embryo, stay armed."""
    sentinel_every: int | None = 10
    """Run anyway every Nth frame while disarmed. None disables the backstop."""
    fallback_after_s: float | None = None
    """Arm unconditionally once this much wall-clock has elapsed since T0."""

    ALWAYS: "ArmingPolicy" = None  # type: ignore[assignment]  # set below


ArmingPolicy.ALWAYS = ArmingPolicy(predicate=lambda _s: True, sentinel_every=None)


@dataclass(frozen=True)
class Task(Generic[ObsT]):
    """A perception objective: how to see, how to read, how to score.

    Stage classification and every onset detector are both ``Task`` instances.
    A ``Solver`` (existing) is one way to build ``run`` for the stage task; a
    ``SignalSpec`` (below) is one way to build ``run`` for an onset task.
    """

    name: str
    render: RenderSpec
    run: TaskRun[ObsT]
    obs_type: type[ObsT]
    scorer: Callable[..., Any]
    """Scoring lives in harness/eval/; core/ only carries an opaque reference."""
    arming: ArmingPolicy = ArmingPolicy.ALWAYS
    on_observe: Callable[[ObsT], tuple[Action, ...]] = lambda _o: ()
    """Per-task action hook — e.g. emit SetCadence on first onset."""


# ==========================================================================
# Q1 — generic signal-onset spec
# ==========================================================================

@dataclass(frozen=True)
class SignalSpec:
    """Declarative onset detector. ``make_onset_task(spec) -> Task`` lives in
    ``harness/tasks/onset/_spec.py`` and supplies the two-call run + scorer.

    The dopaminergic detector is one ~30-line instance of this; hatching is
    another. Nothing here is dat-1-specific.
    """

    name: str
    describe_prompt: str
    """Perceiver: vision call → free-form prose about the signal of interest."""
    rubric: Mapping[Intensity, str]
    """Classifier: text-only call mapping prose → Intensity per these criteria."""
    threshold: Intensity = Intensity.WEAK
    """First level that counts as 'signal present'."""
    render: RenderSpec = RenderSpec("dual_view_calib")
    arming: ArmingPolicy = ArmingPolicy.ALWAYS
    on_detect: Action | None = None
    """Emitted once, on the first frame at or above threshold."""


# ==========================================================================
# ROUTE — which tasks run this frame
# ==========================================================================

@dataclass(frozen=True, slots=True)
class FrameState:
    """Immutable snapshot of everything known about an embryo so far.

    This is what routers and arming predicates read. Rebuilt each frame from
    accumulated observations — never mutated in place.
    """

    embryo_id: str
    timepoint: int
    elapsed_s: float
    cadence_s: float
    stage_estimate: Stage | None
    """Latest StageObservation.value, or None before the first stage call."""
    armed: frozenset[str]
    """Task names whose ArmingPolicy has latched open."""
    by_task: Mapping[str, tuple[ObservationBase, ...]]
    """Full per-task history. Immutable view."""
    tokens_spent: int = 0


@dataclass(frozen=True)
class TokenBudget:
    per_frame: int | None = None
    per_run: int | None = None
    spent: int = 0

    def affords(self, est: int) -> bool:
        return self.per_frame is None or (self.spent + est) <= self.per_frame


@dataclass(frozen=True)
class TaskInvocation:
    task: str
    why: Literal["always", "armed", "sentinel", "fallback", "triage"]
    """Recorded so eval can attribute misses/waste to a gating path."""
    use_shared_description: bool = False
    """If True, skip the task's own vision call and consume the router's
    shared perceiver prose instead."""


@runtime_checkable
class Router(Protocol):
    """Decides which tasks run this frame, and how.

    Three reference implementations in ``harness/routers/``:
      - always_on        — every registered task, every frame (cost ceiling)
      - gated            — honor ArmingPolicy only
      - shared_perceiver — one vision describe → N text-only classifiers
    """

    name: str

    def select(
        self, state: FrameState, tasks: Mapping[str, Task], budget: TokenBudget
    ) -> tuple[TaskInvocation, ...]: ...

    async def describe(self, frame: TaskInput) -> str | None:
        """Optional shared-perceiver pass. Default routers return None."""
        ...


# ==========================================================================
# POLICY — observations → actions (kept out of loop.py for testability)
# ==========================================================================

Policy = Callable[[tuple[Observation, ...], FrameState], tuple[Action, ...]]


def default_policy(obs: tuple[Observation, ...], _state: FrameState) -> tuple[Action, ...]:
    """Collect per-task ``on_observe`` actions. The microscope-facing policy in
    production may add global rules (e.g. Halt on hatched)."""
    # body intentionally omitted in sketch
    return ()


# ==========================================================================
# RESULT — what one loop step returns
# ==========================================================================

@dataclass(frozen=True)
class PerceptionResult:
    """Output of ``core.loop.step()`` for one frame across all invoked tasks."""

    state: FrameState
    observations: tuple[Observation, ...]
    actions: tuple[Action, ...]
    trajectories: Mapping[str, Trajectory]
    """task name → agent transcript, for the HTML report."""
    invocations: tuple[TaskInvocation, ...]
    """What the router chose and why — drives gating metrics."""


# Scoring types (Score, Scorer, RouterScore) live in harness/eval/, not here —
# core/ must stay free of any GT reference per the grep invariant.
