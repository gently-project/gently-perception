"""
The outer loop over frames.

Owns per-embryo session state (history of predicted Observations) and builds
the FrameInput for each frame. Ground truth never enters this module.
"""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from pathlib import Path

from harness.core import agent, render, verify
from harness.core.solver import Solver
from harness.core.types import FrameInput, ModelOutputError, Observation, Prediction, Stage, now
from harness.io.events import Event

PREV_FRAMES_KEPT = 3
HISTORY_SHOWN = 3


def _fmt_history(hist: tuple[Observation, ...]) -> str:
    if not hist:
        return ""
    lines = [f"T{o.timepoint}: {o.stage.value}" for o in hist[-HISTORY_SHOWN:]]
    return "Recent observations for this embryo:\n" + "\n".join(lines)


def build_frame(
    embryo_id: str,
    timepoint: int,
    volume_path: Path,
    *,
    refs: Mapping[Stage, tuple[str, ...]],
    history: tuple[Observation, ...],
    prev_paths: tuple[Path, ...],
    image_b64: str | None = None,
) -> FrameInput:
    """Construct the immutable FrameInput. Harness-only; solvers never call this."""
    return FrameInput(
        embryo_id=embryo_id,
        timepoint=timepoint,
        image_b64=image_b64 if image_b64 is not None else render.cached_render(volume_path),
        volume_ref=Path(volume_path),
        references=refs,
        history=history,
        history_text=_fmt_history(history),
        last_stage=history[-1].stage if history else None,
        prev_volume_refs=prev_paths,
    )


async def step(
    frame: FrameInput,
    solver: Solver,
    *,
    verifier: verify.Verifier = verify.monotonic,
    on_event: Callable[[Event], None] = agent._noop,
) -> Prediction:
    """Run one frame through the agent loop + verifier. Shared by eval and production."""
    on_event(Event.frame_start(frame.embryo_id, frame.timepoint))
    pred, traj = await agent.react(frame, solver, on_event)
    corrected, reason = verifier(frame, pred)
    if reason:
        on_event(
            Event.verify_override(
                frame.embryo_id, frame.timepoint, pred.stage.value, corrected.stage.value, reason
            )
        )
    on_event(
        Event.prediction(
            frame.embryo_id,
            frame.timepoint,
            corrected.stage.value,
            corrected.reasoning,
            traj.budget_exhausted,
        )
    )
    on_event(Event.frame_end(frame.embryo_id, frame.timepoint, len(traj.steps), traj.total_tokens))
    return corrected


async def _run_embryo(
    embryo_id: str,
    frames: list[tuple[int, Path]],
    solver: Solver,
    *,
    refs: Mapping[Stage, tuple[str, ...]],
    verifier: verify.Verifier,
    on_event: Callable[[Event], None],
) -> list[tuple[FrameInput, Prediction]]:
    """Sequential per-embryo loop — history must accumulate frame by frame."""
    history: list[Observation] = []
    prev_paths: list[Path] = []
    results: list[tuple[FrameInput, Prediction]] = []
    for timepoint, volume_path in frames:
        hist = tuple(history)
        pp = tuple(reversed(prev_paths[-PREV_FRAMES_KEPT:]))
        frame = build_frame(embryo_id, timepoint, volume_path, refs=refs, history=hist, prev_paths=pp)
        try:
            pred = await step(frame, solver, verifier=verifier, on_event=on_event)
        except ModelOutputError as e:
            on_event(Event.error(embryo_id, timepoint, str(e)))
            prev_paths.append(volume_path)
            continue
        history.append(Observation(timepoint, pred.stage, now()))
        prev_paths.append(volume_path)
        results.append((frame, pred))
    return results


async def run_loop(
    source: Iterable[tuple[str, int, Path]],
    solver: Solver,
    *,
    refs: Mapping[Stage, tuple[str, ...]],
    verifier: verify.Verifier = verify.monotonic,
    on_event: Callable[[Event], None] = agent._noop,
    concurrency: int = 1,
) -> AsyncIterator[tuple[FrameInput, Prediction]]:
    """Iterate frames in order, maintaining per-embryo predicted history.

    `source` yields (embryo_id, timepoint, volume_path) — never ground truth.
    Embryo sessions are independent, so with concurrency>1 they run in parallel
    (each embryo's frames stay sequential — history must accumulate in order).
    """
    import asyncio

    by_embryo: dict[str, list[tuple[int, Path]]] = {}
    for embryo_id, timepoint, volume_path in source:
        by_embryo.setdefault(embryo_id, []).append((timepoint, volume_path))

    if concurrency <= 1:
        for embryo_id, frames in by_embryo.items():
            for fr, pred in await _run_embryo(
                embryo_id, frames, solver, refs=refs, verifier=verifier, on_event=on_event
            ):
                yield fr, pred
        return

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(eid: str, frames: list) -> list:
        async with sem:
            return await _run_embryo(eid, frames, solver, refs=refs, verifier=verifier, on_event=on_event)

    tasks = [asyncio.create_task(_bounded(eid, fr)) for eid, fr in by_embryo.items()]
    for results in await asyncio.gather(*tasks):
        for fr, pred in results:
            yield fr, pred
