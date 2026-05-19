"""Outer loop: history is predictions-only; events emitted in order; verify clamps."""
import numpy as np
import pytest

from harness.core import loop, verify
from harness.core.solver import Solver
from harness.core.types import Stage
from harness.io.events import Event
from tests.conftest import arun, classify_block, fake_response


@pytest.fixture
def tiny_source(tmp_path):
    paths = []
    for e in (1, 2):
        for t in (0, 1, 2):
            p = tmp_path / f"embryo_{e}_T{t:03d}.npz"
            np.savez(p, np.random.rand(8, 16, 16).astype(np.float32))
            paths.append((f"embryo_{e}", t, p))
    return paths


def test_history_is_predictions_only(tiny_source, stub_generate, monkeypatch):
    """Each frame's history equals prior predictions for that embryo — never GT."""
    s = Solver(name="t", system="sys", tools=(), max_steps=1)
    for stage in ["early", "bean", "comma", "early", "early", "bean"]:
        stub_generate.queue.append(fake_response([classify_block(stage)]))

    seen: dict[tuple[str, int], tuple] = {}
    orig = loop.build_frame

    def spy(eid, t, vp, **kw):
        seen[(eid, t)] = tuple(o.stage for o in kw["history"])
        return orig(eid, t, vp, **kw)

    monkeypatch.setattr(loop, "build_frame", spy)
    events: list[Event] = []
    arun(loop.run_loop(tiny_source, s, refs={}, on_event=events.append))

    assert seen[("embryo_1", 2)] == (Stage.EARLY, Stage.BEAN)
    assert seen[("embryo_2", 0)] == ()
    assert "ground_truth" not in str(events)


def test_events_emitted_in_order(tiny_source, stub_generate):
    s = Solver(name="t", system="sys", tools=(), max_steps=1)
    for _ in range(6):
        stub_generate.queue.append(fake_response([classify_block("early")]))
    events: list[Event] = []
    arun(loop.run_loop(tiny_source, s, refs={}, on_event=events.append))
    kinds = [e.kind for e in events]
    assert kinds[:4] == ["frame_start", "model_turn", "prediction", "frame_end"]
    assert kinds.count("prediction") == 6


def test_verify_monotonic_clamps(tiny_source, stub_generate):
    s = Solver(name="t", system="sys", tools=(), max_steps=1)
    for stage in ["pretzel", "early", "early", "early", "early", "early"]:
        stub_generate.queue.append(fake_response([classify_block(stage)]))
    events: list[Event] = []
    results = arun(loop.run_loop(tiny_source, s, refs={}, verifier=verify.monotonic, on_event=events.append))
    preds = [(f.embryo_id, f.timepoint, p.stage) for f, p in results]
    assert preds[1] == ("embryo_1", 1, Stage.PRETZEL)
    assert any(e.kind == "verify_override" for e in events)
