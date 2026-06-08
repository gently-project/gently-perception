"""Rotated views demoted to 1.5fold/2fold tie-breaker (fix for pretzel undercount).

hybrid_3dviews told the model the true fold count is the max seen across
rotated views. Tight pretzel coils merge in ANY max-projection, so every view
of an early pretzel shows ~2 bands and the model downgraded pretzel→2fold
(28 frames lost at the embryo_8 transition, pretzel 59%→54%).

Same images, inverted authority: eggshell-fill density stays the primary
pretzel discriminator; rotated views are consulted ONLY to split 1.5fold from
2fold, where the true count is ≤2 and band counting is reliable. A low
segment count in rotated views is explicitly NOT evidence against pretzel.
"""
from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage
from harness.solvers.hybrid_3dviews import ANGLES
from harness.solvers.hybrid_nodefer import (
    _SCIENTIFIC_ANALYSIS,
    _TEMPORAL_ANALYSIS,
    SCIENTIFIC_SYSTEM as _SCI_BASE,
    TEMPORAL_SYSTEM as _TMP_BASE,
)
from harness.tools.rotate import rotated_mip_b64

_3D_SECTION = """\

## ROTATED VIEWS — 1.5FOLD vs 2FOLD ONLY

After the main 3-view image you are given the SAME volume rotated about the \
embryo's long axis by 45°, 90°, and 135°, then max-projected. These views \
have ONE job — separating 1.5fold from 2fold:

- TWO clearly separated parallel bright bands in ANY view → 2fold.
- Every view shows one continuous band with at most a partial fold → 1.5fold.

They CANNOT rule out pretzel: tightly coiled bodies MERGE into 2-3 broad \
bands in every projection, so a low segment count in the rotated views is \
NOT evidence against pretzel. Decide pretzel from eggshell fill density in \
the main views, exactly as you would without the rotated views.
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
    """hybrid_3dviews block structure with the tie-breaker framing in the tail text."""
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
            f"then max-projection): {', '.join(f'{a}°' for a in ANGLES)} in order. "
            "Use them only for the 1.5fold/2fold distinction."
        )
    )
    for angle in ANGLES:
        blocks.append(model.image_block(rotated_mip_b64(frame.volume_ref, angle)))
    base = _SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS
    blocks.append(model.text_block(base))
    return blocks


SOLVER = Solver(name="hybrid_3dviews2", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
