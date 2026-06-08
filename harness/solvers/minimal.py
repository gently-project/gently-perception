"""Minimal one-shot baseline — stage names only, no descriptions."""
from harness.core.solver import Solver

# Prompt verbatim from perception/minimal.py for parity.
from perception.minimal import SYSTEM_PROMPT as SYSTEM

SOLVER = Solver(name="minimal", system=SYSTEM, tools=(), max_steps=1)
