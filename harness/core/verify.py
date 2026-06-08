"""
Post-prediction verification — a harness phase, not solver logic.

Verifiers take (frame, prediction) and return a possibly-corrected prediction.
The loop emits a verify_override event when a correction happens.
"""
from __future__ import annotations

from collections.abc import Callable

from harness.core.types import FrameInput, Prediction, Stage

Verifier = Callable[[FrameInput, Prediction], tuple[Prediction, str | None]]


def none(frame: FrameInput, pred: Prediction) -> tuple[Prediction, str | None]:
    return pred, None


def monotonic(frame: FrameInput, pred: Prediction) -> tuple[Prediction, str | None]:
    """Clamp predictions that jump >1 stage backward from the last observation.

    Embryo development is monotonic; a >1-stage regression is almost always a
    perception error, not biology.
    """
    if frame.last_stage is None:
        return pred, None
    delta = pred.stage.ordinal - frame.last_stage.ordinal
    if delta < -1:
        corrected = Prediction(
            stage=frame.last_stage,
            reasoning=f"[verify:monotonic] clamped {pred.stage.value}→{frame.last_stage.value}; original: {pred.reasoning}",
            raw=pred.raw,
        )
        return corrected, f"regressed {abs(delta)} stages"
    return pred, None


VERIFIERS: dict[str, Verifier] = {"none": none, "monotonic": monotonic}
