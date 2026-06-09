"""hybrid_annot2 + raymarched views at the annotator's own poses.

Third attempt at giving the model 3D shape information, with both failure
causes of the MIP attempts addressed:
- hybrid_3dviews/3dviews2 used max-projections, where tight pretzel coils
  merge into bands; these are true raymarched renders with depth occlusion —
  the same viewer output the human annotator labeled from.
- Those attempts also instructed the model to re-decide stages from the new
  views; here the views are framed as supporting evidence for the existing
  tail-progress criteria (find the tail tip), with the annot2 ruleset
  unchanged on top.

Two views per frame (annotator default pose + his tail-visibility pose),
~50ms each cached.

RESULT (embryos 5-8, claude-opus-4-6, 3 seeds): exact 66.5% +/- 4.0
(seeds 68.4/69.2/62.0), adjacent 85.2% — worse than hybrid_annot2's
69.5 +/- 0.2 and the highest variance of the annot line. 2fold dropped
26 -> 17.5 and pretzel destabilized (65.3 +/- 9.7). Third negative result
for extra static views; with projection-collapse ruled out by the
raymarcher, the remaining explanation is that additional static images
dilute attention rather than add usable evidence on this task. Kept in
the registry for reference; hybrid_annot2 remains the production solver.
"""
from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage
from pathlib import Path

from harness.solvers.hybrid_annot import _SCIENTIFIC_ANALYSIS, _TEMPORAL_ANALYSIS
from harness.solvers.hybrid_annot2 import (
    build_systems,
    frame_interval_minutes,
    pretzel_window_text,
)
from harness.tools.annotator_view import annotator_view_b64

_3D_SECTION = """\

## 3D VIEWER RENDERS

After the main 3-view projection you are given two renders from the same 3D \
viewer the human annotator used to label this data: first his default \
working pose, then a side pose he used to identify the tail ("you can see \
the two lobes clearly — the left is the tail"). Unlike the flat projections, \
these have depth — nearer structure occludes farther structure, so separate \
body folds stay visually separate.

Use them to apply the tail criteria: find the tail (the shorter lobe), and \
judge how far its tip has progressed. The projections remain your primary \
evidence; the renders are there to resolve where the tail is when the \
projection is ambiguous.
"""

def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    interval = frame_interval_minutes(str(Path(frame.volume_ref).parent))
    tmp, sci = build_systems(interval)
    base = sci if _is_scientific(frame) else tmp
    out = base.replace("\nRespond with JSON:", _3D_SECTION + "\nRespond with JSON:")
    assert out != base, "3D section insertion point missing"
    return out


def _user_blocks(frame: FrameInput) -> list[dict]:
    """annot2's structure + the two raymarched views before the analysis prompt."""
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
    blocks.append(
        model.text_block(
            "3D viewer renders (annotator's default pose, then his tail-visibility pose):"
        )
    )
    blocks.append(model.image_block(annotator_view_b64(frame.volume_ref, "default")))
    blocks.append(model.image_block(annotator_view_b64(frame.volume_ref, "tail")))
    blocks.append(model.text_block(_SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS))
    return blocks


SOLVER = Solver(name="hybrid_annotviews", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
