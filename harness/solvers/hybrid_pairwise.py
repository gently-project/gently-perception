"""hybrid_annot2 + temporal pairing: the previous frame's image in every turn.

Every remaining failure cluster is transition-shaped: the model lags
boundaries by ~5 frames and skips brief stages entirely. All week it has had
only TEXT history ("T059: 2fold") — it has never been able to LOOK at the
previous frame. This makes "did the boundary just get crossed?" a direct
visual comparison instead of a memory exercise.

Two additions over hybrid_annot2 (documented together; ablate if it wins):
1. The previous frame's standard 3-view projection rendered immediately
   before the current frame's, labeled, with the analysis tail asking
   "what changed between these two images?" first.
2. The contrastive boundary descriptions (from hybrid_contrast3d, derived
   from embryos 2-3 — no eval leakage) as the vocabulary for naming the
   change. Flat under viewer autonomy, but they were written for exactly
   this comparison question.

No tools, one-shot, same cadence-aware ruleset — any delta is attributable
to seeing the previous frame (plus its vocabulary).

RESULT (embryos 5-8, claude-opus-4-6, 3 seeds): exact 68.9% +/- 1.3
(seed0 70.2 — first seed of any experiment above annot2's level),
adjacent 87.0 — statistical tie with hybrid_annot2 (69.5 +/- 0.2).
The mechanistic readout is the real finding: the failure clusters are
IDENTICAL to annot2's (behind 655 vs 641, lag median 8 vs 9, windows
missed 35 vs 33). Seeing the previous frame did not reduce the lag at
all — the model holds the old stage even with both frames side by side
and the boundary's first-visible-change description in hand. The lag is
an under-updating DECISION behavior, not a perception gap; consistent
with every image-side intervention failing while anchor-removal and
forced-update rules (nodefer, the timing rule) produced every win.
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
from harness.solvers.hybrid_contrast3d import BOUNDARIES_SECTION

_PAIR_SECTION = """\

## PREVIOUS-FRAME COMPARISON

You are shown the PREVIOUS frame's projection directly above the current \
one. Compare them before anything else. Most frame pairs show NO \
transition — the embryo simply persists in its stage, possibly shifted by \
twitching. Your first question is: did the FIRST VISIBLE CHANGE of any \
boundary (see the stage-transition list) appear between these two images? \
If nothing structural changed, keep the previous stage. If something did, \
name which boundary it is and advance exactly one stage.
"""


def _prev_volume_ref(frame: FrameInput) -> Path | None:
    """Previous timepoint's volume: timepoint indexes the sorted file list."""
    if frame.timepoint == 0:
        return None
    files = sorted(Path(frame.volume_ref).parent.glob("*.tif"))
    idx = frame.timepoint - 1
    if idx >= len(files) or files[frame.timepoint] != Path(frame.volume_ref):
        return None  # index/file mismatch — skip the pair rather than mislead
    return files[idx]


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    interval = frame_interval_minutes(str(Path(frame.volume_ref).parent))
    tmp, sci = build_systems(interval)
    base = sci if _is_scientific(frame) else tmp
    out = base.replace(
        "\nRespond with JSON:", BOUNDARIES_SECTION + _PAIR_SECTION + "\nRespond with JSON:"
    )
    assert out != base, "insertion point missing"
    return out


def _user_blocks(frame: FrameInput) -> list[dict]:
    """annot2's structure + the previous frame's image before the current one."""
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
        blocks.append(model.text_block(f"PREVIOUS frame (T{frame.timepoint - 1}):"))
        blocks.append(model.image_block(cached_render(prev_ref)))
    blocks.append(model.text_block(f"CURRENT frame (T{frame.timepoint}):"))
    blocks.append(model.image_block(frame.image_b64))
    base = _SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS
    blocks.append(
        model.text_block(
            "First: compare the two frames — did any boundary's first visible "
            "change appear between them? Then " + base[0].lower() + base[1:]
        )
    )
    return blocks


SOLVER = Solver(name="hybrid_pairwise", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
