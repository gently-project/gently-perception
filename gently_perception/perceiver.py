"""Core perceiver: session management and the call loop.

The Perceiver wraps any perceive function with per-embryo session management.
It accumulates context through sequential calls — each call caches the image
and records the observation, so the perceive function (and its tools) can
look back at previous timepoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from .examples import load_examples
from .types import PerceptionOutput


@dataclass
class Observation:
    """A single perception observation for an embryo."""

    timepoint: int
    timestamp: datetime
    stage: str
    reasoning: str


class Session:
    """Per-embryo observation history and image cache."""

    def __init__(self, embryo_id: str):
        self.embryo_id = embryo_id
        self.observations: list[Observation] = []
        self.images: dict[int, str] = {}  # timepoint -> image_b64

    @property
    def current_stage(self) -> str | None:
        return self.observations[-1].stage if self.observations else None

    @property
    def history(self) -> list[dict]:
        """Last N observations as dicts for prompt injection."""
        return [
            {"timepoint": o.timepoint, "stage": o.stage}
            for o in self.observations[-5:]
        ]

    @property
    def stability(self) -> int:
        """Consecutive observations at the current stage."""
        if not self.observations:
            return 0
        current = self.observations[-1].stage
        count = 0
        for o in reversed(self.observations):
            if o.stage == current:
                count += 1
            else:
                break
        return count

    def get_image(self, timepoint: int) -> str | None:
        """Get a cached image by timepoint (for tools)."""
        return self.images.get(timepoint)

    def get_previous_image(self, current_timepoint: int, offset: int = 1) -> tuple[int, str] | None:
        """Get an image from N timepoints back."""
        available = sorted(tp for tp in self.images if tp < current_timepoint)
        if len(available) >= offset:
            tp = available[-offset]
            return tp, self.images[tp]
        return None

    def summary(self) -> dict[str, Any]:
        """Rich summary for external reasoning layers."""
        from .temporal import analyze_temporal
        from .organism import CELEGANS

        return {
            "embryo_id": self.embryo_id,
            "current_stage": self.current_stage,
            "stability": self.stability,
            "observation_count": len(self.observations),
            "stage_sequence": [o.stage for o in self.observations[-10:]],
            "temporal": analyze_temporal(
                self.observations, CELEGANS.stage_durations
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize session state (for persistence across restarts)."""
        return {
            "embryo_id": self.embryo_id,
            "observations": [
                {
                    "timepoint": o.timepoint,
                    "timestamp": o.timestamp.isoformat(),
                    "stage": o.stage,
                    "reasoning": o.reasoning,
                }
                for o in self.observations
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        """Restore session from serialized data."""
        session = cls(data["embryo_id"])
        for obs_data in data.get("observations", []):
            session.observations.append(
                Observation(
                    timepoint=obs_data["timepoint"],
                    timestamp=datetime.fromisoformat(obs_data["timestamp"]),
                    stage=obs_data["stage"],
                    reasoning=obs_data.get("reasoning", ""),
                )
            )
        return session


class Perceiver:
    """Wraps any perceive function with per-embryo session management.

    The Perceiver is self-contained: it loads its own reference examples,
    manages per-embryo sessions, and caches images for tool access.

    Usage::

        perceiver = Perceiver()
        result = await perceiver("embryo_1", 0, image_b64, timestamp)
        result = await perceiver("embryo_1", 1, image_b64, timestamp)
        # perceiver accumulates context through calls
    """

    def __init__(
        self,
        perceive_fn: Callable | None = None,
        references: dict[str, list[str]] | None = None,
    ):
        if perceive_fn is None:
            perceive_fn = _default_perceive_fn()
        self.perceive_fn = perceive_fn
        self.references = references if references is not None else load_examples()
        self.sessions: dict[str, Session] = {}

    async def __call__(
        self,
        embryo_id: str,
        timepoint: int,
        image_b64: str,
        timestamp: datetime,
        volume=None,
    ) -> PerceptionOutput:
        session = self.sessions.setdefault(embryo_id, Session(embryo_id))
        session.images[timepoint] = image_b64

        result = await self.perceive_fn(
            image_b64=image_b64,
            references=self.references,
            history=session.history,
            timepoint=timepoint,
            volume=volume,
            session=session,
        )

        session.observations.append(
            Observation(
                timepoint=timepoint,
                timestamp=timestamp,
                stage=result.stage,
                reasoning=result.reasoning,
            )
        )

        return result

    def get_session(self, embryo_id: str) -> Session | None:
        return self.sessions.get(embryo_id)


def _default_perceive_fn():
    """Lazy-load the default (best) perceive function."""
    import importlib
    import sys
    from pathlib import Path

    # Try to import from experiments/prompt/hybrid.py
    experiments_dir = Path(__file__).parent.parent / "experiments"
    if experiments_dir.exists():
        sys.path.insert(0, str(experiments_dir))
        try:
            mod = importlib.import_module("prompt.hybrid")
            return mod.perceive_hybrid
        except (ImportError, AttributeError):
            pass
        finally:
            if str(experiments_dir) in sys.path:
                sys.path.remove(str(experiments_dir))

    raise ImportError(
        "No default perceive function found. "
        "Either pass perceive_fn= to Perceiver() or ensure "
        "experiments/prompt/hybrid.py exists."
    )
