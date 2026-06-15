"""
Onset-task scoring.

Reads an events.jsonl, joins predictions against ``OnsetGroundTruth``, and
reports detection latency, false-positive rate, and miss rate. The router
metrics (arm_lead, wasted_frames) are computed here too once invocations are
recorded; for v1 (no router) they're left at zero.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from harness.core.types_v2 import Intensity, IntensityObservation
from harness.io.events import read_events


@dataclass(frozen=True)
class Score:
    primary: float
    breakdown: dict[str, float] = field(default_factory=dict)

from harness.io.ground_truth_onset import OnsetGroundTruth


@dataclass
class OnsetRunScore:
    n: int = 0
    n_errors: int = 0
    per_embryo: dict[str, dict] = field(default_factory=dict)
    tokens: int = 0

    @property
    def detected(self) -> list[str]:
        return [e for e, r in self.per_embryo.items() if r.get("detected_at") is not None]

    @property
    def miss_rate(self) -> float:
        pos = [e for e, r in self.per_embryo.items() if r["gt_onset"] is not None]
        if not pos:
            return 0.0
        missed = [e for e in pos if self.per_embryo[e].get("detected_at") is None]
        return len(missed) / len(pos)

    @property
    def mean_latency(self) -> float | None:
        lats = [
            r["detected_at"] - r["gt_onset"]
            for r in self.per_embryo.values()
            if r["gt_onset"] is not None and r.get("detected_at") is not None
        ]
        return sum(lats) / len(lats) if lats else None

    @property
    def fp_rate(self) -> float:
        neg_frames = sum(r["n_neg_frames"] for r in self.per_embryo.values())
        fps = sum(r["n_false_pos"] for r in self.per_embryo.values())
        return fps / neg_frames if neg_frames else 0.0

    def summary(self) -> dict:
        return {
            "n_frames": self.n,
            "n_errors": self.n_errors,
            "miss_rate": round(self.miss_rate, 3),
            "mean_latency_frames": self.mean_latency,
            "fp_rate": round(self.fp_rate, 3),
            "tokens": self.tokens,
            "per_embryo": self.per_embryo,
        }


def score_onset_run(
    events_path: Path, gt: OnsetGroundTruth, *, threshold: Intensity
) -> OnsetRunScore:
    s = OnsetRunScore()
    for eid in gt.embryo_ids:
        s.per_embryo[eid] = {
            "gt_onset": gt.onset.get(eid),
            "detected_at": None,
            "n_neg_frames": 0,
            "n_false_pos": 0,
            "trace": [],
        }

    for ev in read_events(events_path):
        if ev.kind == "error":
            s.n_errors += 1
            continue
        if ev.kind == "frame_end":
            s.tokens += int(ev.payload.get("tokens", 0))
            continue
        if ev.kind != "prediction":
            continue
        eid, tp = ev.embryo_id, ev.timepoint
        if eid not in s.per_embryo:
            continue
        s.n += 1
        level = Intensity(ev.payload["value"])
        row = s.per_embryo[eid]
        row["trace"].append((tp, level.value))
        is_pos_gt = gt.is_positive(eid, tp)
        called_pos = level >= threshold
        if is_pos_gt is False:
            row["n_neg_frames"] += 1
            if called_pos:
                row["n_false_pos"] += 1
        if called_pos and row["detected_at"] is None:
            row["detected_at"] = tp
    return s


def latency_scorer(threshold: Intensity):
    """Factory for a Task.scorer — wraps score_onset_run into the Score shape."""

    def _score(
        obs: Sequence[IntensityObservation], gt: OnsetGroundTruth, invocations
    ) -> Score:
        # In v1 we score from events.jsonl, not from in-memory obs; this adapter
        # exists so Task.scorer is populated. runner_task calls score_onset_run
        # directly.
        return Score(primary=0.0, breakdown={})

    return _score
