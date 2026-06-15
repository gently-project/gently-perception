"""
Single-task runner (v1 — no router).

Runs one ``Task`` over its dataset's volume stream, writes events.jsonl in the
same shape as the solver runner, and scores with ``score_onset``. Embryos run
in parallel; frames within an embryo run sequentially so own_history
accumulates.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
import time
from pathlib import Path

from harness.core.types_v2 import FrameState, IntensityObservation, ModelOutputError, TaskInput
from harness.eval.score_onset import score_onset_run
from harness.io.datasets import DATASETS
from harness.io.events import Event, EventWriter
from harness.io.ground_truth_onset import OnsetGroundTruth
from harness.io.volumes import OfflineSource
from harness.render import render as render_spec
from harness.tasks import get_task

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def _run_dir(task_name: str, model: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    d = RUNS_DIR / f"task-{task_name}" / model / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def _group_by_embryo(source: OfflineSource, allow: set[str] | None):
    by: dict[str, list[tuple[int, Path]]] = {}
    for eid, tp, path in source:
        if allow and eid not in allow:
            continue
        by.setdefault(eid, []).append((tp, path))
    return by


async def _run_embryo(task, eid: str, frames: list[tuple[int, Path]], emit) -> None:
    history: list[IntensityObservation] = []
    for tp, vol_path in frames:
        emit(Event.frame_start(eid, tp))
        try:
            img = render_spec(task.render.name, vol_path, task.render.params)
            inp = TaskInput(
                embryo_id=eid,
                timepoint=tp,
                timestamp=time.time(),
                image_b64=img,
                volume_ref=vol_path,
                own_history=tuple(history),
                state=FrameState(
                    embryo_id=eid,
                    timepoint=tp,
                    elapsed_s=0.0,
                    cadence_s=0.0,
                    stage_estimate=None,
                    armed=frozenset({task.name}),
                    by_task={task.name: tuple(history)},
                ),
            )
            obs, traj = await task.run(inp)
            history.append(obs)
            emit(
                Event(
                    kind="prediction",
                    ts=time.time(),
                    embryo_id=eid,
                    timepoint=tp,
                    payload={
                        "task": task.name,
                        "value": obs.value.value,
                        "onset": obs.onset,
                        "reasoning": obs.reasoning,
                        "description": obs.raw.get("description", ""),
                    },
                )
            )
            emit(Event.frame_end(eid, tp, n_steps=len(traj.steps), tokens=traj.total_tokens))
        except ModelOutputError as e:
            emit(Event.error(eid, tp, str(e)))
            emit(Event.frame_end(eid, tp, n_steps=0, tokens=0))


async def run_task(
    task_name: str,
    *,
    n_runs: int = 1,
    embryos: set[str] | None = None,
    limit: int | None = None,
    concurrency: int = 4,
    volumes_dir: Path | None = None,
    gt_path: Path | None = None,
) -> dict:
    task = get_task(task_name)
    mod = importlib.import_module(f"harness.tasks.onset.{task_name}")
    ds = DATASETS[getattr(mod, "DATASET")]

    vdir = volumes_dir or ds.volumes_dir
    if not vdir.exists():
        print(
            f"error: volumes dir {vdir} not found.\n"
            f"Run: python setup_data.py --dataset {ds.name}",
            file=sys.stderr,
        )
        sys.exit(3)

    source = OfflineSource(vdir)
    allow = embryos or (set(ds.embryo_filter) or None)
    by_embryo = _group_by_embryo(source, allow)
    if limit:
        by_embryo = {e: fs[:limit] for e, fs in by_embryo.items()}
    if not by_embryo:
        print(f"error: no frames found in {vdir} for embryos={allow}", file=sys.stderr)
        sys.exit(3)

    from harness.core.model import DEFAULT_MODEL

    run_dir = _run_dir(task_name, DEFAULT_MODEL)
    print(f"→ {run_dir}")

    gt = OnsetGroundTruth.from_json(gt_path or ds.onset_gt)
    from harness.tasks._base import spec_sha

    seed_scores = []
    for seed in range(n_runs):
        seed_dir = run_dir / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        events_path = seed_dir / "events.jsonl"
        with EventWriter(events_path) as emit:
            emit(
                Event.run_start(
                    {
                        "task": task_name,
                        "dataset": ds.name,
                        "model": DEFAULT_MODEL,
                        "spec_sha": spec_sha(mod.SPEC),
                        "gt_sha": gt.sha(),
                        "n_embryos": len(by_embryo),
                        "n_frames": sum(len(v) for v in by_embryo.values()),
                    }
                )
            )
            sem = asyncio.Semaphore(concurrency)

            async def one(eid, frames):
                async with sem:
                    await _run_embryo(task, eid, frames, emit)

            await asyncio.gather(*(one(e, f) for e, f in by_embryo.items()))
            score = score_onset_run(events_path, gt, threshold=mod.SPEC.threshold)
            emit(Event.run_end(score.summary()))
        seed_scores.append(score)
        print(f"  seed{seed}: {json.dumps(score.summary(), default=str)}")

    agg = {
        "run_dir": str(run_dir),
        "miss_rate": [s.miss_rate for s in seed_scores],
        "mean_latency": [s.mean_latency for s in seed_scores],
        "fp_rate": [s.fp_rate for s in seed_scores],
    }
    (run_dir / "summary.json").write_text(json.dumps(agg, indent=2, default=str))
    print(json.dumps(agg, indent=2, default=str))
    return agg
