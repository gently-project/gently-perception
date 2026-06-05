"""hybrid_annot + late-pretzel vs hatching disambiguation.

Targets the one remaining 'ahead' failure cluster: 33 consecutive end-of-series
frames on embryo_5 (x3 seeds = 99 failed predictions) where a very late,
shell-filling, vigorously moving pretzel was called "hatching", and history
anchoring then locked the streak in ("it has been hatching for 2 frames").

Two-part fix, both prompts:
1. Duration argument kills the trigger — hatching is a near-instant event
   (rupture, exit, empty shell within a frame or two), while late pretzel
   presses against and deforms the shell and wriggles WITHIN it for a long
   time, which mimics emergence.
2. Explicit cascade breaker — a multi-frame "hatching" streak in the history
   is self-contradictory; treat it as evidence the earlier calls were wrong
   rather than something to continue.

The annotator's embryo_5 notes back the first part: "the worm moves a lot ...
within the egg shell" on exactly these frames.
"""
from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage
from harness.solvers.hybrid_annot import (
    _SCIENTIFIC_ANALYSIS,
    _TEMPORAL_ANALYSIS,
    SCIENTIFIC_SYSTEM as _SCI_BASE,
    TEMPORAL_SYSTEM as _TMP_BASE,
)

_OLD_TMP_RULE = """\
5. **Late pretzel.** The pretzel stage is long-lasting. A compact bright mass \
that fills the eggshell is still pretzel even if it doesn't look "tangled" — \
it has not hatched unless you see the worm OUTSIDE the shell."""

_NEW_TMP_RULE = """\
5. **Late pretzel vs hatching.** The pretzel stage is long-lasting, and near \
its end the worm moves vigorously WITHIN the eggshell — pressing against it, \
deforming it, and changing pose between frames. This mimics emergence but is \
still pretzel. Hatching, by contrast, is nearly instantaneous: the shell \
ruptures, the worm exits, and within a frame or two you see a thin worm \
clearly OUTSIDE the shell or an empty/partial shell. If the bright mass is \
still shell-sized and shell-shaped, nothing has hatched. A history showing \
several consecutive "hatching" observations is self-contradictory — hatching \
completes almost immediately, so a long streak means those earlier calls \
were wrong; re-examine against the shell outline instead of continuing the \
streak."""

_OLD_SCI_TEXT = """\
**HATCHING/HATCHED**: The worm is emerging or has left the eggshell — a thin \
elongated worm OUTSIDE the eggshell boundary, or an empty shell."""

_NEW_SCI_TEXT = """\
**HATCHING/HATCHED**: The worm is emerging or has left the eggshell — a thin \
elongated worm OUTSIDE the eggshell boundary, or an empty shell. Hatching is \
nearly instantaneous (rupture → exit → empty shell within a frame or two). \
Late pretzel moves vigorously WITHIN the shell, pressing against and \
deforming it — that mimics emergence but is still pretzel as long as the \
bright mass stays shell-sized and shell-shaped. A multi-frame "hatching" \
streak in the history is self-contradictory; treat it as evidence the \
earlier calls were wrong rather than something to continue."""

TEMPORAL_SYSTEM = _TMP_BASE.replace(_OLD_TMP_RULE, _NEW_TMP_RULE)
SCIENTIFIC_SYSTEM = _SCI_BASE.replace(_OLD_SCI_TEXT, _NEW_SCI_TEXT)
assert TEMPORAL_SYSTEM != _TMP_BASE, "temporal late-pretzel rule did not match"
assert SCIENTIFIC_SYSTEM != _SCI_BASE, "scientific hatching text did not match"


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches harness/solvers/hybrid.py."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    return SCIENTIFIC_SYSTEM if _is_scientific(frame) else TEMPORAL_SYSTEM


def _user_blocks(frame: FrameInput) -> list[dict]:
    """Identical to hybrid_annot."""
    blocks: list[dict] = [model.text_block(f"\n=== CLASSIFY EMBRYO AT T{frame.timepoint} ===")]
    if frame.history_text:
        blocks.append(model.text_block(frame.history_text))
        last = frame.last_stage.value if frame.last_stage else "early"
        blocks.append(
            model.text_block(
                f"The most recent observation was '{last}'. "
                f"When uncertain, prefer the earlier stage — except at the "
                f"2fold→pretzel boundary: once the history shows ~10-15 frames of "
                f"2fold, prolonged 2fold appearance plus any twisting reads as "
                f"pretzel, per the timing rule."
            )
        )
    blocks.append(model.image_block(frame.image_b64))
    blocks.append(model.text_block(_SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS))
    return blocks


SOLVER = Solver(name="hybrid_annot2", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
