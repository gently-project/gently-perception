"""Solver registry — explicit dict, not auto-discovery."""
from harness.core.solver import Solver
from harness.solvers import agentic_baseline, hybrid, hybrid_nodefer, hybrid_noprior, minimal, scientific, temporal

REGISTRY: dict[str, Solver] = {
    "minimal": minimal.SOLVER,
    "temporal": temporal.SOLVER,
    "scientific": scientific.SOLVER,
    "hybrid": hybrid.SOLVER,
    "hybrid_nodefer": hybrid_nodefer.SOLVER,
    "hybrid_noprior": hybrid_noprior.SOLVER,
    "agentic_baseline": agentic_baseline.SOLVER,
}
