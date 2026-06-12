"""Full 3D viewer autonomy + contrastive transition-boundary descriptions.

hybrid_explorer3d (rotate/zoom/click-to-center, 8-step budget, eggshell
guard) scored 67.1 +/- 1.1 with the dominant failures unchanged: every
remaining cluster is a transition-DETECTION failure — the model recognizes
stages but misses the frame where one ends and the next begins.

This adds boundary descriptions written contrastively ("what visibly
changes"), derived frame-by-frame from the ground-truth transition frames of
embryos 2-3 (session 59799c78 — the other annotated session, so no eval-set
leakage). Each names the first visible change at the boundary rather than
describing the stages on either side.

Comparisons: vs hybrid_explorer3d isolates the contrastive text within the
autonomy arm; vs hybrid_annot2 (69.5, text-only) is a two-variable
comparison — if this wins, a text-only contrastive arm separates the
contributions.

RESULT (embryos 5-8, claude-opus-4-6, 3 seeds): exact 67.0% +/- 1.6,
adjacent 86.4 — a statistical tie with hybrid_explorer3d (67.1 +/- 1.1)
and still -2.5 vs text-only hybrid_annot2 (69.5 +/- 0.2). The boundary
descriptions did not move their target stages (comma ~5%, bean ~29%,
1.5fold ~14%, 2fold ~18% — unchanged from every annot-line run), and
added nothing under autonomy overall. Whether they help on a text-only
base remains untested.
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
from harness.solvers.hybrid_explorer3d import _TOOL_SECTION

BOUNDARIES_SECTION = """\

## STAGE TRANSITIONS — THE FIRST VISIBLE CHANGE AT EACH BOUNDARY

Most classification errors are missed transitions, not misrecognized \
stages. Compare the current frame against the recent history and ask which \
boundary, if any, has just been crossed:

- **early → bean: the oval loses its symmetry.** A perfect, smooth, convex \
bright oval is early. The transition is NOT a new structure — it is one \
corner of the outline pulling slightly inward (the future posterior \
indentation) and a faint dim seam appearing inside the mass. If the outline \
is no longer a clean symmetric oval, bean has begun.

- **bean → comma: the seam becomes a notch.** Bean is a "peanut": two \
roughly equal bright lobes with a shallow dim line between them. When that \
line deepens into a dark WEDGE cutting in from the edge — real dark space \
opening between a distinct smaller lobe (the tail) and the body — comma has \
begun. Peanut → hook.

- **comma → 1.5fold: the first parallel-band pair appears.** Comma is still \
ONE bright mass with a notch and a tail lobe. The transition is a \
reorganization of the whole image: TWO parallel bright bands separated by a \
long dark diagonal furrow (body axis + folded-back tail), the tail band \
clearly shorter, reaching about halfway. Counting goes from "one mass" to \
"two bands" — the most categorical change of the series.

- **1.5fold → 2fold: the short band catches up.** Band count stays TWO — do \
not wait for a third. In 1.5fold the tail band is short, leaving a large \
dark region at one end of the eggshell. When both bands run nearly the full \
long axis (the embryo reads as an S or zigzag) and the dark space has shrunk \
to two thin furrows, 2fold has begun. The cue is relative band LENGTH, not \
band count.

- **2fold → pretzel: the dark furrows fragment.** In 2fold the dark spaces \
between body segments are LONG, CONTINUOUS channels — an organized zigzag. \
At pretzel the coils cross and the continuous furrows break into SMALL, \
ISOLATED dark pockets; texture turns granular/marbled and the outline fills \
almost completely. Judge the CONTINUITY of the dark spaces: long channels = \
2fold, scattered pockets = pretzel. This works even when you cannot count \
coils.
"""


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    interval = frame_interval_minutes(str(Path(frame.volume_ref).parent))
    tmp, sci = build_systems(interval)
    base = sci if _is_scientific(frame) else tmp
    out = base.replace(
        "\nRespond with JSON:", BOUNDARIES_SECTION + _TOOL_SECTION + "\nRespond with JSON:"
    )
    assert out != base, "insertion point missing"
    return out


def _user_blocks(frame: FrameInput) -> list[dict]:
    """Identical to hybrid_explorer3d — 3D views arrive via tool calls."""
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
    name="hybrid_contrast3d",
    system=_system_for,
    tools=("view3d",),
    max_steps=8,
    user_blocks=_user_blocks,
)
