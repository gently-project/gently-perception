"""
Production entry point — thin Perceiver wrapping the same core.loop.step()
the eval runner uses. Ships frozen on the hybrid solver.
"""
from __future__ import annotations

from pathlib import Path

from harness.core import loop, verify
from harness.core.solver import Solver
from harness.core.types import Observation, Prediction, Stage, now
from harness.solvers import hybrid


class Perceiver:
    def __init__(
        self,
        solver: Solver = hybrid.SOLVER,
        *,
        verifier: verify.Verifier = verify.monotonic,
        refs: dict[Stage, tuple[str, ...]] | None = None,
    ) -> None:
        self._solver = solver
        self._verifier = verifier
        self._refs = refs or {}
        self._sessions: dict[str, list[Observation]] = {}
        self._prev_paths: dict[str, list[Path]] = {}

    async def __call__(
        self, embryo_id: str, timepoint: int, *, image_b64: str, volume_path: Path | None = None
    ) -> Prediction:
        hist = tuple(self._sessions.get(embryo_id, []))
        pp = tuple(reversed(self._prev_paths.get(embryo_id, [])[-loop.PREV_FRAMES_KEPT:]))
        frame = loop.build_frame(
            embryo_id,
            timepoint,
            volume_path or Path("/nonexistent"),
            refs=self._refs,
            history=hist,
            prev_paths=pp,
            image_b64=image_b64,
        )
        pred = await loop.step(frame, self._solver, verifier=self._verifier)
        self._sessions.setdefault(embryo_id, []).append(Observation(timepoint, pred.stage, now()))
        if volume_path:
            self._prev_paths.setdefault(embryo_id, []).append(volume_path)
        return pred
