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
- measure: quantitative features (convexity, n_segments, aspect_ratio)

Use tools when the default view is ambiguous — especially for the fold stages \
where counting overlapping bands is hard. measure(convexity) DECREASES as the \
body folds: ~0.80 early/bean/comma, ~0.76 at 1.5fold, ~0.65 at 2fold, ~0.57 at \
pretzel, ~0.32 hatched. The 2fold and pretzel ranges overlap — use the number \
as supporting evidence alongside what you see, never as the deciding factor on \
its own.

When confident, call classify_stage with your answer. You do not need to use \
every tool or exhaust your tool budget — classify as soon as the evidence is \
clear. Stages change slowly: if recent history shows stage X, the current frame \
is very likely also X or X+1.
"""

# thinking must stay off: the API rejects thinking when tool_choice forces tool
# use, and the agent loop forces tool choice on every step (that forcing is what
# guarantees a classify_stage call and therefore no silent failure). This also
# keeps the experiment clean — baseline is (no tools, no thinking), so the
# agentic arm is (tools, no thinking): the tool effect is isolated.
SOLVER = Solver(
    name="agentic_baseline",
    system=SYSTEM,
    tools=("zoom", "z_slice", "z_sweep", "prev_frame", "measure"),
    max_steps=5,
)
