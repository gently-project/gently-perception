"""
Contrastive perception function (stub).

Single API call with transition-based descriptions — instead of describing
each stage in isolation, describes the *differences* between adjacent stages.
Tests whether contrastive framing reduces boundary confusion.

TODO: implement when ready to run this experiment.
"""

from ._base import PerceptionOutput


async def perceive_contrastive(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    raise NotImplementedError(
        "perceive_contrastive is a stub — implement when ready to run this experiment"
    )
