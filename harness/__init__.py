"""
Agentic perception harness — eval-first, single source of truth.

See harness/CLAUDE.md for the autonomous-agent iteration loop.
"""
from harness.core.solver import Solver
from harness.core.types import FrameInput, Observation, Prediction, Stage, Trajectory
from harness.perceiver import Perceiver
from harness.tools import tool

__all__ = ["FrameInput", "Observation", "Perceiver", "Prediction", "Solver", "Stage", "Trajectory", "tool"]
