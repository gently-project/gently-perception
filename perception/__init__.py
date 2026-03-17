"""
Modular perception functions for VLM-based embryo stage classification.

Each function has the same async signature::

    async def perceive(
        image_b64: str,
        references: dict[str, list[str]],
        history: list[dict],
        timepoint: int,
    ) -> PerceptionOutput

The FUNCTIONS registry maps variant names to their perceive() callables.
This is the code the agent modifies — equivalent to train.py in autoresearch.
"""

from ._base import PerceptionOutput  # noqa: F401

# Lazy registry — populated on first access via get_functions()
_FUNCTIONS: dict | None = None


def get_functions() -> dict:
    """Return the registry mapping variant name -> perceive callable."""
    global _FUNCTIONS
    if _FUNCTIONS is not None:
        return _FUNCTIONS

    from .minimal import perceive_minimal
    from .descriptive import perceive_descriptive
    from .minimal_multishot import perceive_minimal_multishot
    from .descriptive_multishot import perceive_descriptive_multishot

    _FUNCTIONS = {
        "minimal": perceive_minimal,
        "descriptive": perceive_descriptive,
        "minimal_multishot": perceive_minimal_multishot,
        "descriptive_multishot": perceive_descriptive_multishot,
    }
    return _FUNCTIONS
