"""End-to-end smoke test for the onset task pipeline.

Exercises render → perceive → classify → events → score with a synthetic
volume and mocked model calls. No API key or downloaded data required.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from harness.core.types_v2 import Intensity
from harness.eval.score_onset import score_onset_run
from harness.io.ground_truth_onset import OnsetGroundTruth


@pytest.fixture
def fake_volumes(tmp_path: Path) -> Path:
    """Six tiny .npz volumes for one embryo. load_volume() reads npz['volume']."""
    vdir = tmp_path / "vols" / "embryo_3"
    vdir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(6):
        vol = (rng.random((4, 32, 32)) * 200).astype(np.uint16)
        np.savez(vdir / f"embryo_3_2026_01_01_{i:02d}.npz", volume=vol)
    return tmp_path / "vols"


@pytest.fixture
def fake_gt(tmp_path: Path) -> Path:
    p = tmp_path / "gt.json"
    p.write_text(json.dumps({"onset": {"embryo_3": 3}}))
    return p


def _msg(text: str | None = None, tool_input: dict | None = None):
    """Minimal stand-in for anthropic.types.Message."""
    content = []
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    if tool_input is not None:
        content.append(
            SimpleNamespace(type="tool_use", name="classify_intensity", input=tool_input)
        )
    return SimpleNamespace(
        content=content, usage=SimpleNamespace(input_tokens=10, output_tokens=5)
    )


def test_onset_pipeline_end_to_end(fake_volumes: Path, fake_gt: Path, tmp_path: Path):
    """Signal absent for tp 0-2, present from tp 3. Expect detection at tp 3,
    zero false positives, latency 0."""

    levels = ["none", "none", "none", "weak", "medium", "strong"]
    call_idx = {"n": 0}

    async def fake_generate(**kw):
        # alternates perceiver (no tools) / classifier (forced tool)
        if not kw.get("tools"):
            return _msg(text="frame description")
        i = call_idx["n"]
        call_idx["n"] += 1
        return _msg(tool_input={"intensity_level": levels[i], "reasoning": "test"})

    with (
        mock.patch("harness.tasks._base.generate", side_effect=fake_generate),
        mock.patch("harness.eval.runner_task.RUNS_DIR", tmp_path / "runs"),
    ):
        from harness.eval.runner_task import run_task

        agg = asyncio.run(
            run_task(
                "dopaminergic",
                n_runs=1,
                embryos={"embryo_3"},
                volumes_dir=fake_volumes,
                gt_path=fake_gt,
            )
        )

    run_dir = Path(agg["run_dir"])
    events = run_dir / "seed0" / "events.jsonl"
    assert events.exists()

    from harness.tasks.onset.dopaminergic import SPEC

    gt = OnsetGroundTruth.from_json(fake_gt)
    score = score_onset_run(events, gt, threshold=SPEC.threshold)
    row = score.per_embryo["embryo_3"]

    assert row["detected_at"] == 3, row
    assert row["n_false_pos"] == 0, row
    assert score.miss_rate == 0.0
    assert score.mean_latency == 0.0
    assert score.fp_rate == 0.0
    assert score.n == 6
    assert score.tokens > 0


def test_onset_gt_negative_control():
    gt = OnsetGroundTruth(onset={"embryo_1": None, "embryo_3": 5})
    assert gt.is_positive("embryo_1", 100) is False
    assert gt.is_positive("embryo_3", 4) is False
    assert gt.is_positive("embryo_3", 5) is True
    assert gt.is_positive("embryo_9", 0) is None
