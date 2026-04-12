"""Shared types for the perception harness."""

from dataclasses import dataclass


@dataclass
class PerceptionOutput:
    """What every perceive function returns."""

    stage: str
    reasoning: str
    raw_response: str = ""
