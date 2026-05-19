"""ReAct loop: dispatch, classify-stops, budget-forces-classify, no-tool-use raises."""
import pytest

from harness.core.agent import react
from harness.core.solver import Solver
from harness.core.types import ModelOutputError, Stage
from tests.conftest import arun, classify_block, fake_response, tool_block


def test_one_shot_forces_classify(tiny_frame, stub_generate):
    s = Solver(name="t", system="sys", tools=(), max_steps=1)
    stub_generate.queue.append(fake_response([classify_block("comma")]))
    pred, traj = arun(react(tiny_frame, s))
    assert pred.stage == Stage.COMMA
    assert traj.n_tool_calls == 0
    assert stub_generate.calls[0]["tool_choice"] == {"type": "tool", "name": "classify_stage"}


def test_agentic_dispatch_then_classify(tiny_frame, stub_generate):
    s = Solver(name="t", system="sys", tools=("measure",), max_steps=3)
    stub_generate.queue.append(fake_response([tool_block("measure", feature="fill_fraction")]))
    stub_generate.queue.append(fake_response([classify_block("2fold")]))
    pred, traj = arun(react(tiny_frame, s))
    assert pred.stage == Stage.TWO_FOLD
    assert traj.n_tool_calls == 1
    assert traj.steps[1].tool_name == "measure"
    assert stub_generate.calls[0]["tool_choice"] == {"type": "any"}
    assert not traj.budget_exhausted


def test_budget_exhaustion_forces_classify(tiny_frame, stub_generate):
    s = Solver(name="t", system="sys", tools=("measure",), max_steps=2)
    stub_generate.queue.append(fake_response([tool_block("measure", feature="aspect_ratio")]))
    stub_generate.queue.append(fake_response([tool_block("measure", feature="fill_fraction")]))
    stub_generate.queue.append(fake_response([classify_block("pretzel")]))
    pred, traj = arun(react(tiny_frame, s))
    assert pred.stage == Stage.PRETZEL
    assert traj.budget_exhausted
    assert stub_generate.calls[-1]["tool_choice"] == {"type": "tool", "name": "classify_stage"}


def test_no_tool_use_raises(tiny_frame, stub_generate):
    s = Solver(name="t", system="sys", tools=(), max_steps=1)
    stub_generate.queue.append(fake_response([{"type": "text", "text": "I think it's comma"}]))
    with pytest.raises(ModelOutputError):
        arun(react(tiny_frame, s))


def test_tool_error_fed_back(tiny_frame, stub_generate):
    """Out-of-range param → ErrorResult fed back as is_error tool_result."""
    s = Solver(name="t", system="sys", tools=("z_slice",), max_steps=3)
    stub_generate.queue.append(fake_response([tool_block("z_slice", index=999)]))
    stub_generate.queue.append(fake_response([classify_block("early")]))
    pred, traj = arun(react(tiny_frame, s))
    assert pred.stage == Stage.EARLY
    assert traj.steps[1].output_kind == "error"
    tool_result_msg = stub_generate.calls[1]["messages"][-1]["content"][0]
    assert tool_result_msg["is_error"]
