"""Solver registry — explicit dict, not auto-discovery."""
from harness.core.solver import Solver
from harness.solvers import agentic_baseline, hybrid, hybrid_3dviews, hybrid_3dviews2, hybrid_agentic3d, hybrid_contrast3d, hybrid_explorer3d, hybrid_annot, hybrid_annot2, hybrid_annotviews, hybrid_nodefer, hybrid_noprior, hybrid_pairwise, minimal, scientific, temporal

REGISTRY: dict[str, Solver] = {
    "minimal": minimal.SOLVER,
    "temporal": temporal.SOLVER,
    "scientific": scientific.SOLVER,
    "hybrid": hybrid.SOLVER,
    "hybrid_nodefer": hybrid_nodefer.SOLVER,
    "hybrid_annot": hybrid_annot.SOLVER,
    "hybrid_annot2": hybrid_annot2.SOLVER,
    "hybrid_annotviews": hybrid_annotviews.SOLVER,
    "hybrid_agentic3d": hybrid_agentic3d.SOLVER,
    "hybrid_explorer3d": hybrid_explorer3d.SOLVER,
    "hybrid_contrast3d": hybrid_contrast3d.SOLVER,
    "hybrid_pairwise": hybrid_pairwise.SOLVER,
    "hybrid_3dviews": hybrid_3dviews.SOLVER,
    "hybrid_3dviews2": hybrid_3dviews2.SOLVER,
    "hybrid_noprior": hybrid_noprior.SOLVER,
    "agentic_baseline": agentic_baseline.SOLVER,
}
