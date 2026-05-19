"""Solver registry — explicit dict, not auto-discovery."""
from harness.core.solver import Solver
from harness.solvers import agentic_baseline, hybrid, minimal, scientific, temporal

REGISTRY: dict[str, Solver] = {
    "minimal": minimal.SOLVER,
    "temporal": temporal.SOLVER,
    "scientific": scientific.SOLVER,
    "hybrid": hybrid.SOLVER,
    "agentic_baseline": agentic_baseline.SOLVER,
}
