"""
Aggregate N seed scores → mean±std table, tool-usage stats, trajectory viewer.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path

from harness.core.types import STAGE_ORDER, Stage
from harness.eval.score import RunScore
from harness.io.events import read_events


def aggregate(scores: list[RunScore]) -> dict:
    """Mean ± std of accuracy across seeds, plus per-stage breakdown."""
    accs = [s.accuracy for s in scores]
    adj = [s.adjacent_accuracy for s in scores]
    per_stage: dict[Stage, list[float]] = defaultdict(list)
    for s in scores:
        for stage, (hits, n) in s.per_stage.items():
            if n:
                per_stage[stage].append(hits / n)
    return {
        "n_runs": len(scores),
        "n": scores[0].n if scores else 0,
        "accuracy_mean": statistics.mean(accs) if accs else 0.0,
        "accuracy_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
        "adjacent_mean": statistics.mean(adj) if adj else 0.0,
        "per_stage": {
            s.value: (statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0.0, len(v))
            for s, v in per_stage.items()
        },
        "tool_calls": dict(sum((s.tool_calls for s in scores), start=type(scores[0].tool_calls)())) if scores else {},
        "mean_steps": statistics.mean([s.mean_steps for s in scores]) if scores else 0.0,
        "n_budget_exhausted": sum(s.n_budget_exhausted for s in scores),
        "n_errors": sum(s.n_errors for s in scores),
    }


def print_table(agg: dict, *, baseline_mean: float | None = None, baseline_stale: bool = False) -> None:
    print()
    print(f"{'=' * 60}")
    print(f"  n={agg['n']}  runs={agg['n_runs']}")
    print(f"  exact:    {agg['accuracy_mean']:.1%} ± {agg['accuracy_std']:.1%}")
    print(f"  adjacent: {agg['adjacent_mean']:.1%}")
    if baseline_mean is not None:
        tag = " (stale)" if baseline_stale else ""
        delta = agg["accuracy_mean"] - baseline_mean
        print(f"  baseline: {baseline_mean:.1%}{tag}  Δ={delta:+.1%}")
    print(f"{'-' * 60}")
    print(f"  {'stage':<12} {'acc':>8} {'±std':>8}")
    for stage in STAGE_ORDER:
        if stage.value in agg["per_stage"]:
            m, s, _ = agg["per_stage"][stage.value]
            print(f"  {stage.value:<12} {m:>7.1%} {s:>7.1%}")
    if agg["tool_calls"]:
        print(f"{'-' * 60}")
        print(f"  tools: {dict(agg['tool_calls'])}")
        print(f"  mean_steps={agg['mean_steps']:.1f}  budget_exhausted={agg['n_budget_exhausted']}")
    if agg["n_errors"]:
        print(f"  ⚠ errors: {agg['n_errors']}")
    print(f"{'=' * 60}\n")


def view_trajectory(events_path: Path, embryo_id: str, timepoint: int) -> None:
    """Print the full agent conversation for one frame."""
    print(f"\n── Trajectory: {embryo_id} T{timepoint} ──")
    for ev in read_events(events_path):
        if ev.embryo_id != embryo_id or ev.timepoint != timepoint:
            continue
        match ev.kind:
            case "model_turn":
                p = ev.payload
                print(f"  [step {ev.step}] model: in={p.get('input_tokens')} out={p.get('output_tokens')} cache={p.get('cache_read')}")
            case "tool_call":
                print(f"  [step {ev.step}] tool {ev.tool_name}({ev.payload.get('params')}) → {ev.payload.get('result_kind')}")
            case "prediction":
                print(f"  ⇒ {ev.payload['stage']}: {ev.payload.get('reasoning', '')[:120]}")
            case "verify_override":
                print(f"  ⚠ verify: {ev.payload['original']} → {ev.payload['corrected']} ({ev.payload['reason']})")
            case "error":
                print(f"  ✗ error: {ev.payload['message']}")
    print()
