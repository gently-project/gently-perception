"""
Dopaminergic-neuron onset detector — strain SLS762, dat-1p::mNeonGreen.

Prompts ported from gently/app/detectors/dopaminergic_signal.py @ 36d4fbc.
The reporter turns on around the 3-fold (pretzel) stage; arming opens at
1.5-fold with sentinel sampling so a stage-classifier lag cannot cause a miss.
"""
from __future__ import annotations

from harness.core.types_v2 import (
    ArmingPolicy,
    Intensity,
    RenderSpec,
    SetCadence,
    SignalSpec,
    Stage,
)
from harness.tasks._base import make_onset_task

DESCRIBE_PROMPT = """\
You are looking at the max projection of a volume of C. elegans imaged with a \
488 nm light-sheet microscope. The embryo is expressing a fluorophore that \
lights up when certain neurons are born.

We are trying to image the birth of these neurons and their continued \
existence. Your description will be read by a classifier that decides how to \
guide the imaging: it will speed up imaging once the neuronal structures \
appear. Be specific so the classifier has something concrete to act on.

You may initially see a faint outline of the embryo — autofluorescence from \
the body.

Eventually, you may see puncta-like structures in the embryo region of the \
image. If there are any puncta outside the embryo region, ignore them — those \
are likely gut granules.

The nerve cells, as they begin to express, will first appear as a faint blob, \
then a brighter blob, and will start to emit thread-like structures from them \
— the nerve body. These are what we want to image.

The embryo may also eventually hatch. Mention it if the embryo structure \
disappears from the field of view.

Describe what you see in a few sentences of plain prose."""

RUBRIC = {
    Intensity.NONE: "no puncta in the embryo / blank / nothing above background.",
    Intensity.WEAK: (
        "one dim spot, OR signal explicitly described as barely visible / "
        "could be noise / very faint."
    ),
    Intensity.MEDIUM: "two or more clearly discrete bright spots above background.",
    Intensity.STRONG: "multiple bright, well-resolved spots; neurite traces may be mentioned.",
    Intensity.SATURATING: "signal explicitly saturates the camera.",
}

SPEC = SignalSpec(
    name="dopaminergic",
    describe_prompt=DESCRIBE_PROMPT,
    rubric=RUBRIC,
    threshold=Intensity.MEDIUM,
    render=RenderSpec("single_view_fixed", {"lo": 100.0, "hi": 4000.0}),
    arming=ArmingPolicy(
        predicate=lambda s: (s.stage_estimate or Stage.EARLY).ordinal >= Stage.ONE_HALF_FOLD.ordinal,
        latch=True,
        sentinel_every=10,
        fallback_after_s=6 * 3600,
    ),
    on_detect=SetCadence(interval_s=60, reason="dopaminergic onset"),
)

TASK = make_onset_task(SPEC)
DATASET = "sls762"
