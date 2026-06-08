"""Scientific one-shot — eggshell-fill-fraction discriminator."""
from harness.core.solver import Solver

from perception.scientific import SYSTEM_PROMPT as SYSTEM

SOLVER = Solver(name="scientific", system=SYSTEM, tools=(), max_steps=1)
