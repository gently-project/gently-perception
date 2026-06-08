"""hybrid_nodefer + static rotated 3D views for body-shape classification.

Adds three extra max-intensity projections of the volume rotated about the
anterior-posterior axis (45°/90°/135°) to every user turn, with prompt
guidance to count body folds across views. Folds that overlap in the default
XY projection separate under rotation — exactly the signal the weak stages
(comma/1.5fold/2fold, all fold-count confusions) are missing.

One-shot like hybrid_nodefer so the 3D-information effect is isolated from
agent-loop variance. Rendering is content-addressed-cached (pre-warm with
do_not_commit/prewarm_rotations.py before the first eval, or the first seed
pays ~0.7s × 3 angles per frame).
"""
from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage
from harness.solvers.hybrid_nodefer import (
    _SCIENTIFIC_ANALYSIS,
    _TEMPORAL_ANALYSIS,
    SCIENTIFIC_SYSTEM as _SCI_BASE,
    TEMPORAL_SYSTEM as _TMP_BASE,
)
from harness.tools.rotate import rotated_mip_b64

ANGLES = (45, 90, 135)

_3D_SECTION = """\

## 3D BODY SHAPE

After the main 3-view image you are given the SAME volume rotated about the \
embryo's long (anterior-posterior) axis by 45°, 90°, and 135°, then \
max-projected. Use them to read the body's 3D shape:

- A fold that overlaps another in one projection SEPARATES from it in the \
rotated views. Count distinct body segments in EACH view — the true fold \
count is the MAXIMUM seen across views, not what the default view shows.
- A smooth continuous outline in all four views means an unfolded body \
(comma or earlier) even if the default view hints at internal banding.
- Dark gaps between bright bands that persist across several views are real \
inter-fold gaps, not projection artifacts.
"""

TEMPORAL_SYSTEM = _TMP_BASE.replace(
    "\nRespond with JSON:", _3D_SECTION + "\nRespond with JSON:"
)
SCIENTIFIC_SYSTEM = _SCI_BASE.replace(
    "\nRespond with JSON:", _3D_SECTION + "\nRespond with JSON:"
)


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    return SCIENTIFIC_SYSTEM if _is_scientific(frame) else TEMPORAL_SYSTEM


def _user_blocks(frame: FrameInput) -> list[dict]:
    """hybrid_nodefer's structure + the three rotated views before the analysis prompt."""
    blocks: list[dict] = [model.text_block(f"\n=== CLASSIFY EMBRYO AT T{frame.timepoint} ===")]
    if frame.history_text:
        blocks.append(model.text_block(frame.history_text))
        last = frame.last_stage.value if frame.last_stage else "early"
        blocks.append(
            model.text_block(
                f"The most recent observation was '{last}'. "
                f"When uncertain, prefer the earlier stage."
            )
        )
    blocks.append(model.image_block(frame.image_b64))
    blocks.append(
        model.text_block(
            "Rotated views of the same volume (rotation about the body axis, "
            f"then max-projection): {', '.join(f'{a}°' for a in ANGLES)} in order."
        )
    )
    for angle in ANGLES:
        blocks.append(model.image_block(rotated_mip_b64(frame.volume_ref, angle)))
    base = _SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS
    blocks.append(
        model.text_block(base + " Count body folds across ALL four views before classifying.")
    )
    return blocks


SOLVER = Solver(name="hybrid_3dviews", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
