"""Hybrid one-shot — switches prompt based on last observed stage.

The replicated baseline (81.7 ± 2.4% on hard stages, claude-opus-4-6, no thinking).
"""
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage

from perception.hybrid import SCIENTIFIC_SYSTEM, TEMPORAL_SYSTEM


def _system_for(frame: FrameInput) -> str:
    if frame.last_stage in {Stage.ONE_HALF_FOLD, Stage.TWO_FOLD}:
        return SCIENTIFIC_SYSTEM
    return TEMPORAL_SYSTEM


SOLVER = Solver(name="hybrid", system=_system_for, tools=(), max_steps=1)
