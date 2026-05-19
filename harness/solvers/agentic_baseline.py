"""First agentic experiment — all tools, 5-step budget, adaptive thinking.

Re-tests the "tools hurt" finding (23% vs 35% on Sonnet 4.5) with a properly
instrumented harness on Opus 4.6.
"""
from harness.core.solver import Solver

SYSTEM = """\
You are classifying C. elegans embryo developmental stages from fluorescence \
light-sheet microscopy. The first image shows three orthogonal max-intensity \
projections (XY top-left, YZ top-right, XZ bottom-left) of the current frame.

Stages in order: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

You have inspection tools available:
- zoom: crop a region of one projection at full resolution
- z_slice / z_sweep: view individual depth planes (max-projection collapses overlapping folds)
- prev_frame: view an earlier timepoint for comparison
- measure: quantitative features (fill_fraction, n_segments, aspect_ratio)

Use tools when the default view is ambiguous — especially for the fold stages \
where counting overlapping bands is hard. measure(fill_fraction) is the most \
discriminative single feature: ~0.5→1.5fold, ~0.7→2fold, ~0.9→pretzel.

When confident, call classify_stage with your answer. Stages change slowly — \
if recent history shows stage X, the current frame is very likely also X or X+1.
"""

SOLVER = Solver(
    name="agentic_baseline",
    system=SYSTEM,
    tools=("zoom", "z_slice", "z_sweep", "prev_frame", "measure"),
    max_steps=5,
    thinking="adaptive",
    effort="high",
)
