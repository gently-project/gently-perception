"""hybrid_annot2 + full 3D viewer autonomy: rotate, zoom, click-to-center.

Completes the autonomy ladder that hybrid_agentic3d started. That run gave
rotation + threshold only, full-embryo renders at fixed distance, 3 calls
max — and lost to text-only annot2 (64.9 +/- 5.3 vs 69.5 +/- 0.2), partly
because 512px whole-body renders are too coarse to read twist, and partly
because late-pretzel renders were misread as hatched.

This variant gives the model everything the human annotator's viewer had:
- rotate (yaw/pitch) and threshold, as before
- zoom up to 4x around a clicked point (center_x/y_pct), so it can inspect
  the tail tip or a suspected fold crossing at fold scale
- an 8-step budget for a survey -> zoom -> decide workflow

Plus one guard distilled from the agentic3d failure: renders show signal
only, not the eggshell wall, so hatched/hatching judgments stay owned by
the projections and the cadence rules.

RESULT (embryos 5-8, claude-opus-4-6, 3 seeds): exact 67.1% +/- 1.1
(adjacent 85.9) — vs hybrid_annot2 69.5 +/- 0.2 / 87.8. Best-behaved 3D
variant yet: the eggshell guard mostly held (pretzel→hatch 0/0/23 per
seed vs 70 in agentic3d's worst seed) and variance came down 5.3 → 1.1,
but full viewer autonomy (rotate + zoom + click-to-center, 8 steps,
~4 renders per exploring frame) still loses to text-only by 2.4 points.
The dominant confusions are unchanged (pretzel→2fold ~60/seed). Fifth
and most conclusive negative for 3D input on this task: the bottleneck
is boundary calibration, not visual information. hybrid_annot2 remains
the production solver.
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

You can use the same 3D viewer the human annotator used — rotate the \
embryo, zoom in, and recenter, exactly as he did when deciding these \
stages: view3d(yaw_deg, pitch_deg, threshold, zoom_pct, center_x_pct, \
center_y_pct). Renders have depth — nearer structure occludes farther \
structure — so folds that overlap in the flat projections separate as you \
rotate, and zooming (up to 4x, centered on a point you pick from a previous \
render at the same angle) shows fold-scale detail the full-body view \
cannot.

A good workflow when the projections leave you uncertain:
1. Survey: one or two rotations (e.g. yaw 90 side view; yaw 45 pitch 30 \
oblique) to find the angle where the ambiguous feature — tail tip, or a \
suspected fold crossing — is least occluded.
2. Inspect: zoom 200-400% on that feature at that angle; raise threshold \
(50-60) if dim outer signal hides the fold cores.
3. Decide using the tail-progress and twisting criteria.

Two cautions:
- If the projections plus history already support a confident call, answer \
immediately — extra views of an unambiguous frame add noise.
- The renders show fluorescent signal only, NOT the eggshell wall. A worm \
pressed against the shell looks edge-on in 3D exactly like an emerging one. \
Never conclude hatching/hatched from renders — that judgment belongs to the \
projections and the elapsed-time rules.
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
    """Identical to hybrid_annot2 — 3D views arrive via tool calls."""
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
    name="hybrid_explorer3d",
    system=_system_for,
    tools=("view3d",),
    max_steps=8,
    user_blocks=_user_blocks,
)
