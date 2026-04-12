"""Shared types for the perception harness."""

from dataclasses import dataclass


@dataclass
class PerceptionOutput:
    """What every perceive function returns.

    confidence is optional — the paper shows VLM self-reported confidence
    is uncalibrated noise (0.867 correct vs 0.857 wrong). The harness
    derives reliability from session history (stability, temporal analysis)
    rather than this field. Experiments may still populate it for analysis.
    """

    stage: str
    reasoning: str
    confidence: float = 0.0  # VLM self-report; unreliable — see paper
    raw_response: str = ""
