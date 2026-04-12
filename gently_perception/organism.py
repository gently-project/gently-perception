"""Organism configuration — stage definitions as data.

Provides the biological context that the harness uses for temporal
analysis. Currently supports C. elegans; other organisms can be added
as additional OrganismConfig instances.
"""

from dataclasses import dataclass, field


@dataclass
class OrganismConfig:
    """Stage definitions for a model organism."""

    name: str
    stages: list[str]
    stage_durations: dict[str, float] = field(default_factory=dict)  # stage -> typical minutes


CELEGANS = OrganismConfig(
    name="celegans",
    stages=[
        "early", "bean", "comma", "1.5fold",
        "2fold", "pretzel", "hatching", "hatched",
    ],
    stage_durations={
        "early": 350,
        "bean": 20,
        "comma": 30,
        "1.5fold": 30,
        "2fold": 45,
        "pretzel": 300,
        "hatching": 15,
    },
)
