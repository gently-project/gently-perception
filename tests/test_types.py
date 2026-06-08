"""FrameInput is frozen; no GT field exists; Stage enum behaves."""
import dataclasses
import re

import pytest

from harness.core.types import FrameInput, Observation, Prediction, Stage, Trajectory


def test_stage_ordinal_and_distance():
    assert Stage.EARLY.ordinal == 0
    assert Stage.HATCHED.ordinal == 7
    assert Stage.TWO_FOLD.distance(Stage.PRETZEL) == 1
    assert Stage.EARLY.distance(Stage.PRETZEL) == 5


def test_frameinput_is_frozen(tiny_frame):
    with pytest.raises(dataclasses.FrozenInstanceError):
        tiny_frame.timepoint = 99  # type: ignore[misc]


def test_history_is_immutable(tiny_frame):
    assert isinstance(tiny_frame.history, tuple)
    with pytest.raises((TypeError, AttributeError)):
        tiny_frame.history.append(Observation(1, Stage.EARLY, 0.0))  # type: ignore[attr-defined]


def test_no_ground_truth_field():
    """The GT-leak surface cannot exist because FrameInput has no GT field."""
    fields = {f.name for f in dataclasses.fields(FrameInput)}
    assert not any(re.search(r"truth|^gt", f, re.I) for f in fields), fields


def test_prediction_stage_is_enum():
    p = Prediction(stage=Stage.COMMA, reasoning="x", raw={})
    assert isinstance(p.stage, Stage)


def test_trajectory_n_tool_calls():
    from harness.core.types import Step, ImageResult

    t = Trajectory()
    t.steps.append(Step.model(None))
    t.steps.append(Step.tool("zoom", {}, ImageResult(b64="x")))
    t.steps.append(Step.tool("measure", {}, ImageResult(b64="x")))
    assert t.n_tool_calls == 2
