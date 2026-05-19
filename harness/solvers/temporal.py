"""Temporal-anchored one-shot — strong "stages change slowly" prior."""
from harness.core.solver import Solver

from perception.temporal import SYSTEM_PROMPT as SYSTEM

SOLVER = Solver(name="temporal", system=SYSTEM, tools=(), max_steps=1)
