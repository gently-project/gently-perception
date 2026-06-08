"""HTML report generation from a golden events.jsonl — string assertions, no browser."""
import json

from harness.eval.html_report import generate
from harness.io.events import Event, EventWriter


def _make_run(tmp_path, name, preds):
    """preds: list of (timepoint, stage). Returns the run dir."""
    run_dir = tmp_path / name
    with EventWriter(run_dir / "seed0" / "events.jsonl") as w:
        w(Event.run_start({"solver": name, "model": "test-model", "seed": 0}))
        for t, stage in preds:
            w(Event.frame_start("embryo_1", t))
            w(Event.prediction("embryo_1", t, stage, f"reasoning for T{t}", False))
            w(Event.frame_end("embryo_1", t, 1, 100))
        w(Event.run_end({"n_frames": len(preds)}))
    return run_dir


def _gt(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"transitions": {"embryo_1": {"early": 0, "bean": 3, "comma": 6}}}))
    return p


def test_report_generates(tmp_path):
    run = _make_run(tmp_path, "runA", [(0, "early"), (1, "early"), (3, "bean"), (4, "comma"), (6, "comma")])
    out = generate(run, gt_path=_gt(tmp_path), volumes_dir=tmp_path / "novolumes")
    assert out.exists()
    html = out.read_text()
    assert "test-model" in html
    data = json.loads(html.split('type="application/json">')[1].split("</script>")[0])
    assert data["summary"]["n"] == 5
    assert data["summary"]["accuracy_mean"] == 0.8  # t4 wrong
    errs = [f for f in data["frames"] if f["ok"] is False]
    assert len(errs) == 1 and errs[0]["t"] == 4
    assert errs[0]["reasoning"] == "reasoning for T4"


def test_report_failure_clusters(tmp_path):
    """late_arrival / window_missed / ahead assignment, window stats, and example selection."""
    preds = [
        (0, "early"), (1, "early"), (2, "bean"),                  # t2 ahead of gt
        (3, "early"), (4, "early"), (5, "early"),                  # bean window (3-5) never entered
        (6, "bean"), (7, "bean"), (8, "comma"), (9, "comma"),      # comma window (6-9) entered late
    ]
    run = _make_run(tmp_path, "runC", preds)
    out = generate(run, gt_path=_gt(tmp_path), volumes_dir=tmp_path / "novolumes")
    html = out.read_text()
    assert "clustersSec" in html
    data = json.loads(html.split('type="application/json">')[1].split("</script>")[0])
    c = data["clusters"]
    assert c["n_pred"] == 10 and c["n_fail"] == 6
    assert c["behind"] == 5 and c["ahead"] == 1
    assert c["windows_missed"] == 1 and c["windows_total"] == 3
    assert c["lag_median"] == 2 and c["lag_max"] == 2
    by_key = {cl["key"]: cl for cl in c["clusters"]}
    assert by_key["window_missed"]["count"] == 3
    assert by_key["late_arrival"]["count"] == 2
    assert by_key["ahead"]["count"] == 1
    assert by_key["ahead"]["examples"] == [{"e": "embryo_1", "t": 2}]
    assert by_key["late_arrival"]["pairs"][0] == ["comma→bean", 2]


def test_report_compare(tmp_path):
    gt = _gt(tmp_path)
    a = _make_run(tmp_path, "runA", [(0, "early"), (3, "bean"), (4, "comma")])  # t4 wrong
    b = _make_run(tmp_path, "runB", [(0, "early"), (3, "comma"), (4, "bean")])  # t3 wrong, t4 right
    out = generate(a, gt_path=gt, compare_dir=b, volumes_dir=tmp_path / "novolumes")
    data = json.loads(out.read_text().split('type="application/json">')[1].split("</script>")[0])
    cmp = data["compare"]
    assert len(cmp["wins"]) == 1 and cmp["wins"][0]["t"] == 3  # right in A, wrong in B
    assert len(cmp["regressions"]) == 1 and cmp["regressions"][0]["t"] == 4  # wrong in A, right in B


def test_tool_image_externalized(tmp_path):
    """ImageResult in a tool_call event lands in media/ and the JSONL records the path."""
    from harness.core.types import ImageResult, NumericResult

    run_dir = tmp_path / "run"
    with EventWriter(run_dir / "seed0" / "events.jsonl") as w:
        w(Event.run_start({"solver": "x", "model": "m", "seed": 0}))
        w(Event.tool_call("embryo_1", 5, 0, "zoom", {"x": 1}, ImageResult(b64="aGVsbG8=")))
        w(Event.tool_call("embryo_1", 5, 1, "measure", {"feature": "convexity"}, NumericResult(value=0.7)))
        w(Event.prediction("embryo_1", 5, "comma", "r", False))
    media = run_dir / "seed0" / "media" / "embryo_1_T005_s0_zoom.jpg"
    assert media.exists() and media.read_bytes() == b"hello"
    lines = (run_dir / "seed0" / "events.jsonl").read_text().splitlines()
    tool_ev = json.loads(lines[1])
    assert tool_ev["payload"]["image_path"] == "media/embryo_1_T005_s0_zoom.jpg"
    assert "image_b64" not in tool_ev["payload"]
    # Numeric result keeps its value inline
    assert json.loads(lines[2])["payload"]["value"] == 0.7
    # And the report surfaces the step with the seed-relative img path
    out = generate(run_dir, gt_path=_gt(tmp_path), volumes_dir=tmp_path / "novolumes")
    data = json.loads(out.read_text().split('type="application/json">')[1].split("</script>")[0])
    steps = data["frames"][0]["steps"]
    assert steps[0]["img"] == "seed0/media/embryo_1_T005_s0_zoom.jpg"
    assert steps[1]["value"] == 0.7
