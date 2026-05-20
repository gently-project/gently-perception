"""Hybrid one-shot — switches prompt based on last observed stage.

Replicated baseline (81.7±2.4% hard stages, claude-opus-4-6, no thinking).
The user-turn anchoring + trailing analysis prompt are part of the strategy
(perception/hybrid.py:164-208) and are reproduced here verbatim.
"""
from harness.core import model
from harness.core.solver import Solver
from harness.core.types import FrameInput, Stage

from perception.hybrid import SCIENTIFIC_SYSTEM, TEMPORAL_SYSTEM

_SCIENTIFIC_ANALYSIS = (
    "Analyze: (1) How much of the eggshell is filled with signal? "
    "(2) How many parallel body segments are visible? "
    "(3) Which reference images match best? Then classify."
)
_TEMPORAL_ANALYSIS = (
    "Compare this image to the reference images above. Which stage's references "
    "does it most closely match? Classify accordingly."
)


def _is_scientific(frame: FrameInput) -> bool:
    """Scientific prompt for 2fold/pretzel, temporal otherwise. Matches perception/hybrid.py:_get_system_prompt."""
    return frame.last_stage in {Stage.TWO_FOLD, Stage.PRETZEL}


def _system_for(frame: FrameInput) -> str:
    return SCIENTIFIC_SYSTEM if _is_scientific(frame) else TEMPORAL_SYSTEM


def _user_blocks(frame: FrameInput) -> list[dict]:
    """Mirrors perception/hybrid.py:perceive_hybrid user-turn structure."""
    blocks: list[dict] = [model.text_block(f"\n=== CLASSIFY EMBRYO AT T{frame.timepoint} ===")]
    if frame.history_text:
        blocks.append(model.text_block(frame.history_text))
        last = frame.last_stage.value if frame.last_stage else "early"
        blocks.append(
            model.text_block(
                f"The most recent observation was '{last}'. "
                f"Remember: stages change slowly. The current stage is most likely "
                f"'{last}' unless you see a clear morphological change. "
                f"When uncertain, prefer the earlier stage."
            )
        )
    blocks.append(model.image_block(frame.image_b64))
    blocks.append(model.text_block(_SCIENTIFIC_ANALYSIS if _is_scientific(frame) else _TEMPORAL_ANALYSIS))
    return blocks


SOLVER = Solver(name="hybrid", system=_system_for, tools=(), max_steps=1, user_blocks=_user_blocks)
