"""Temporal analysis — pure function over observation history.

Computes how long an embryo has been at its current stage, whether it
might be arrested, and other temporal context. No side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .perceiver import Observation


@dataclass
class TemporalContext:
    """Temporal analysis for an embryo's developmental state."""

    current_stage: str | None
    time_in_stage_min: float
    observations_in_stage: int
    expected_duration_min: float | None
    overtime_ratio: float  # time_in_stage / expected_duration (0 if unknown)
    is_potentially_arrested: bool
    total_observations: int


def analyze_temporal(
    observations: list[Observation],
    stage_durations: dict[str, float],
) -> TemporalContext | None:
    """Compute temporal analysis from observation history.

    Parameters
    ----------
    observations : list[Observation]
        Chronological observations for one embryo.
    stage_durations : dict[str, float]
        Expected duration in minutes per stage (from OrganismConfig).

    Returns
    -------
    TemporalContext or None if no observations.
    """
    if not observations:
        return None

    current = observations[-1]

    # Count consecutive observations at current stage
    obs_in_stage = 0
    for o in reversed(observations):
        if o.stage == current.stage:
            obs_in_stage += 1
        else:
            break

    # Compute time in current stage using timestamps
    stage_start_idx = len(observations) - obs_in_stage
    stage_start_obs = observations[stage_start_idx]
    time_in_stage = (current.timestamp - stage_start_obs.timestamp).total_seconds() / 60.0

    expected = stage_durations.get(current.stage)
    overtime_ratio = time_in_stage / expected if expected and expected > 0 else 0.0

    # Arrest detection: >3x expected duration or >30 observations at same stage
    is_arrested = (
        (expected is not None and time_in_stage > 3 * expected)
        or obs_in_stage > 30
    )

    return TemporalContext(
        current_stage=current.stage,
        time_in_stage_min=time_in_stage,
        observations_in_stage=obs_in_stage,
        expected_duration_min=expected,
        overtime_ratio=overtime_ratio,
        is_potentially_arrested=is_arrested,
        total_observations=len(observations),
    )
