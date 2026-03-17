"""
Hybrid perception function (stub).

Single API call with cherry-picked stage descriptions — only comma and
hatched get detailed descriptions, the rest use names only.  Tests whether
adding descriptions only at confusion-prone boundaries improves accuracy.

TODO: implement when ready to run this experiment.
"""

from ._base import PerceptionOutput


async def perceive_hybrid(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    raise NotImplementedError(
        "perceive_hybrid is a stub — implement when ready to run this experiment"
    )
