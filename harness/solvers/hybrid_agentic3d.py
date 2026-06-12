"""hybrid_annot2 + autonomy to rotate the 3D rendering (view3d tool).

The static-views experiments (hybrid_3dviews, hybrid_3dviews2,
hybrid_annotviews) all regressed: fixed extra images on every frame dilute
attention on the frames that didn't need them. This tests the remaining
hypothesis — that 3D views help when the MODEL decides it needs one and
picks the angle, instead of being handed the same poses every frame.

Same cadence-aware annot2 ruleset; adds the view3d tool (the annotator's
raymarcher with yaw/pitch/threshold control) on a small step budget. The
prompt explicitly licenses answering immediately when the projections
suffice — the agentic_baseline result (80.2 vs 86.2 on embryos 1-4) showed
that forced tool use hurts.

RESULT (embryos 5-8, claude-opus-4-6, 3 seeds): exact 64.9% +/- 5.3
(seeds 69.8/65.5/59.3), adjacent 82.0% — vs hybrid_annot2 69.5 +/- 0.2.
Tool selection behaved as prompted (~33% of frames, ~2.7 calls each,
concentrated on transition frames), but the renders misled at exactly
the boundaries they were meant to resolve: seed2 re-opened the
pretzel→hatched failure (70 frames) that annot2's text rules had fully
fixed — late-pretzel renders show the worm pressed against the shell
and the model reads it as outside. Fourth negative for 3D views
(static MIP x2, static raymarch, agentic raymarch). hybrid_annot2
remains the production solver.
"""
from pathlib import Path

from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage
from harness.solvers.hybrid_annot import _SCIENTIFIC_ANALYSIS, _TEMPORAL_ANALYSIS
from harness.solvers.hybrid_annot2 import (
    build_systems,
    frame_interval_minutes,
    pretzel_window_text,
)

_TOOL_SECTION = """\

## 3D VIEWER (view3d tool)

You can render the volume in the same 3D viewer the human annotator used, \
from any angle: view3d(yaw_deg, pitch_deg, threshold). These renders have \
depth — nearer structure occludes farther structure — so overlapping folds \
separate as you rotate, unlike the flat projections above.

Use it ONLY when the projections leave you genuinely uncertain, typically:
- you cannot tell where the tail tip is (1.5fold vs 2fold) → try a side \
view (yaw 90) or an oblique view (yaw 45, pitch 30) to see the fold plane
- you cannot tell whether the two segments are twisting (2fold vs pretzel) \
→ rotate to look along the body axis, and raise threshold (50-60) to peel \
dim signal off the fold cores

One or two well-chosen views are enough. If the projections plus history \
already support a confident call, answer immediately without the tool — \
extra views of an unambiguous frame add noise, not signal.
"""


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    interval = frame_interval_minutes(str(Path(frame.volume_ref).parent))
    tmp, sci = build_systems(interval)
    base = sci if _is_scientific(frame) else tmp
    out = base.replace("\nRespond with JSON:", _TOOL_SECTION + "\nRespond with JSON:")
    assert out != base, "tool section insertion point missing"
    return out


def _user_blocks(frame: FrameInput) -> list[dict]:
    """Identical to hybrid_annot2 — the 3D views arrive via tool calls instead."""
    blocks: list[dict] = [model.text_block(f"\n=== CLASSIFY EMBRYO AT T{frame.timepoint} ===")]
    if frame.history_text:
        blocks.append(model.text_block(frame.history_text))
        last = frame.last_stage.value if frame.last_stage else "early"
        blocks.append(
            model.text_block(
                f"The most recent observation was '{last}'. "
                f"When uncertain, prefer the earlier stage — except at the "
                f"2fold→pretzel boundary: once the history shows 2fold for "
                f"{pretzel_window_text(frame.volume_ref)}, prolonged 2fold "
                f"appearance plus any twisting reads as pretzel, per the timing rule."
            )
        )
    blocks.append(model.image_block(frame.image_b64))
    blocks.append(model.text_block(_SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS))
    return blocks


SOLVER = Solver(
    name="hybrid_agentic3d",
    system=_system_for,
    tools=("view3d",),
    max_steps=4,
    user_blocks=_user_blocks,
)
