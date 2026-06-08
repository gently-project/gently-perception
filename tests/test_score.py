"""score_run on a golden events.jsonl → known accuracy; --stages filters post-hoc."""
import json

from harness.core.types import Stage
from harness.eval.score import score_run
from harness.io.events import Event, EventWriter
from harness.io.ground_truth import GroundTruth


def make_gt(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"transitions": {"embryo_1": {"early": 0, "bean": 3, "comma": 6}}}))
    return GroundTruth.from_json(p)


def make_events(tmp_path, preds: list[tuple[int, str]]):
    p = tmp_path / "events.jsonl"
    with EventWriter(p) as w:
        for t, stage in preds:
            w(Event.prediction("embryo_1", t, stage, "r", False))
    return p


def test_exact_accuracy(tmp_path):
    gt = make_gt(tmp_path)
    # GT: t0-2=early, t3-5=bean, t6+=comma
    ev = make_events(tmp_path, [(0, "early"), (1, "early"), (3, "bean"), (4, "comma"), (6, "comma")])
    s = score_run(ev, gt)
    assert s.n == 5
    assert s.n_correct == 4  # t4 wrong (bean→comma)
    assert s.accuracy == 0.8


def test_adjacent_accuracy(tmp_path):
    gt = make_gt(tmp_path)
    ev = make_events(tmp_path, [(0, "early"), (3, "comma")])  # t3: bean→comma is adjacent
    s = score_run(ev, gt)
    assert s.adjacent_accuracy == 1.0


def test_stage_filter_post_hoc(tmp_path):
    gt = make_gt(tmp_path)
    ev = make_events(tmp_path, [(0, "early"), (1, "bean"), (3, "bean"), (6, "early")])
    s = score_run(ev, gt, stages={Stage.BEAN})
    assert s.n == 1  # only t3 has GT=bean
    assert s.n_correct == 1


def test_per_stage_breakdown(tmp_path):
    gt = make_gt(tmp_path)
    ev = make_events(tmp_path, [(0, "early"), (1, "early"), (3, "early")])
    s = score_run(ev, gt)
    assert s.per_stage[Stage.EARLY] == (2, 2)
    assert s.per_stage[Stage.BEAN] == (0, 1)
