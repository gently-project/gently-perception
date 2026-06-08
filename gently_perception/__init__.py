"""gently-perception: VLM-based perception harness for microscopy.

The harness provides fixed infrastructure (API calls, session management,
temporal analysis) that perception experiments compose into complete
strategies. Experiments live in ``experiments/`` and are organized by
the dimension of the optimization landscape they explore (prompt,
representation, tools, etc.).

Usage from gently::

    from gently_perception import Perceiver

    perceiver = Perceiver()
    result = await perceiver(embryo_id, timepoint, image_b64, timestamp)

Usage in benchmarks::

    from gently_perception import Perceiver

    perceiver = Perceiver()
    for tc in testcases:
        result = await perceiver(tc.embryo_id, tc.timepoint, tc.image_b64, tc.timestamp)
"""

from .perceiver import Perceiver, Session, Observation
from .types import PerceptionOutput, CameraParams
from .temporal import TemporalContext, analyze_temporal
from .examples import load_examples
from .organism import OrganismConfig, CELEGANS

__all__ = [
    "Perceiver",
    "Session",
    "Observation",
    "PerceptionOutput",
    "CameraParams",
    "TemporalContext",
    "analyze_temporal",
    "load_examples",
    "OrganismConfig",
    "CELEGANS",
]

__version__ = "0.1.0"
