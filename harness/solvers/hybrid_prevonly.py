"""Minimal ablation: hybrid_annot2 + ONLY the previous frame's image.

hybrid_pairwise bundled three changes (previous-frame image, contrastive
boundary section, comparison-first analysis rewrite) and tied annot2 with
wider spread — leaving open whether the image itself helps, hurts, or does
nothing once the text changes are stripped away.

This is annot2 verbatim — same system prompts, same analysis tail, same
user-turn text — plus exactly one thing: the previous frame's projection,
labeled, rendered before the current frame's. Any delta vs annot2
(69.5 +/- 0.2) is attributable to the image alone.

RESULT (embryos 5-8, claude-opus-4-6, 3 seeds): exact 63.7% +/- 1.6,
adjacent 85.0. Paired frame-level bootstrap vs annot2: -5.7 points,
95% CI [-7.2, -4.3] — decisively harmful, where the full pairwise bundle
was a statistical tie (-0.5, CI [-1.6, +0.5]). Confusions shifted
uniformly laggier (pretzel→2fold 72, comma→early appears). Conclusion:
an UNGUIDED previous frame is a visual persistence anchor — consecutive
frames are ~95% identical, and the similarity reads as "same stage"
louder than any boundary cue. pairwise's comparison-first framing wasn't
masking a benefit of the image; it was repairing ~5 of the 6 points of
damage the image causes. Temporal visual context is only safe with
explicit comparison framing.
"""
from pathlib import Path

from harness.core import model
from harness.core.render import cached_render
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage
from harness.solvers.hybrid_annot import _SCIENTIFIC_ANALYSIS, _TEMPORAL_ANALYSIS
from harness.solvers.hybrid_annot2 import (
    build_systems,
    frame_interval_minutes,
    pretzel_window_text,
)
from harness.solvers.hybrid_pairwise import _prev_volume_ref


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    interval = frame_interval_minutes(str(Path(frame.volume_ref).parent))
    tmp, sci = build_systems(interval)
    return sci if _is_scientific(frame) else tmp


def _user_blocks(frame: FrameInput) -> list[dict]:
    """hybrid_annot2's blocks with the labeled previous-frame image inserted."""
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
    prev_ref = _prev_volume_ref(frame)
    if prev_ref is not None:
        blocks.append(model.text_block(f"PREVIOUS frame (T{frame.timepoint - 1}), for reference:"))
        blocks.append(model.image_block(cached_render(prev_ref)))
        blocks.append(model.text_block(f"CURRENT frame (T{frame.timepoint}) — classify this one:"))
    blocks.append(model.image_block(frame.image_b64))
    blocks.append(model.text_block(_SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS))
    return blocks


SOLVER = Solver(name="hybrid_prevonly", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
