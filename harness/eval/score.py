"""
Scoring — reads events.jsonl, joins with ground truth, computes metrics.

This is the ONLY module that imports ground_truth. The --stages filter is
applied here (post-hoc on the event log), not in the agent loop, so GT can
never leak into history.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from harness.core.types import Stage
from harness.io.events import Event, read_events
from harness.io.ground_truth import GroundTruth


@dataclass
class RunScore:
    n: int = 0
    n_correct: int = 0
    n_adjacent: int = 0
    n_errors: int = 0
    per_stage: dict[Stage, tuple[int, int]] = field(default_factory=dict)
    """{stage: (n_correct, n_total)}"""
    confusion: Counter = field(default_factory=Counter)
    """{(true, pred): count}"""
    tool_calls: Counter = field(default_factory=Counter)
    n_budget_exhausted: int = 0
    mean_steps: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    @property
    def adjacent_accuracy(self) -> float:
        return self.n_adjacent / self.n if self.n else 0.0


def score_run(events_path: Path, gt: GroundTruth, *, stages: set[Stage] | None = None) -> RunScore:
    """Compute metrics for one events.jsonl run, optionally filtered to a stage subset."""
    score = RunScore()
    per_stage_hits: dict[Stage, int] = defaultdict(int)
    per_stage_n: dict[Stage, int] = defaultdict(int)
    step_counts: list[int] = []

    for ev in read_events(events_path):
        match ev.kind:
            case "prediction":
                eid, t = ev.embryo_id, ev.timepoint
                assert eid is not None and t is not None
                true = gt.get_stage_at(eid, t)
                if true is None:
                    continue
                if stages and true not in stages:
                    continue
                pred = Stage(ev.payload["stage"])
                score.n += 1
                if pred == true:
                    score.n_correct += 1
                    per_stage_hits[true] += 1
                if pred.distance(true) <= 1:
                    score.n_adjacent += 1
                per_stage_n[true] += 1
                score.confusion[(true.value, pred.value)] += 1
                if ev.payload.get("budget_exhausted"):
                    score.n_budget_exhausted += 1
            case "tool_call":
                if ev.tool_name:
                    score.tool_calls[ev.tool_name] += 1
            case "frame_end":
                step_counts.append(ev.payload.get("n_steps", 0))
            case "error":
                score.n_errors += 1

    score.per_stage = {s: (per_stage_hits[s], per_stage_n[s]) for s in per_stage_n}
    score.mean_steps = sum(step_counts) / len(step_counts) if step_counts else 0.0
    return score


def tool_usage_stats(events: list[Event]) -> dict[str, int]:
    return dict(Counter(e.tool_name for e in events if e.kind == "tool_call" and e.tool_name))
