"""
Ensemble perception function.

Runs the unified prompt 3 times with temperature=0.3 and takes majority
vote. This smooths out stochastic boundary errors where the model
oscillates between adjacent stages.

Cost: 3x API calls per timepoint.
"""

from collections import Counter

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
    STAGES,
)

# Use the unified prompt (best single-prompt design)
from .unified import SYSTEM_PROMPT

ENSEMBLE_SIZE = 3
TEMPERATURE = 0.3


async def perceive_ensemble(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Majority-vote ensemble: run 3x with temperature, take most common stage."""
    content = build_reference_content(references)

    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})
        last_stage = history[-1].get("stage", "unknown") if history else "unknown"
        content.append({
            "type": "text",
            "text": (
                f"The most recent observation was '{last_stage}'. "
                f"Remember: stages change slowly. The current stage is most likely "
                f"'{last_stage}' unless you see a clear morphological change. "
                f"When uncertain, prefer the earlier stage."
            ),
        })

    content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            },
        }
    )

    content.append({
        "type": "text",
        "text": (
            "Analyze step by step: "
            "(1) What fraction of the eggshell is filled with signal — sparse, moderate, or dense? "
            "(2) How many distinct parallel body segments can you count? "
            "(3) Which reference images match best? "
            "Then classify."
        ),
    })

    # Run multiple times with temperature > 0
    import asyncio
    tasks = []
    for _ in range(ENSEMBLE_SIZE):
        tasks.append(
            call_claude(system=SYSTEM_PROMPT, content=content, temperature=TEMPERATURE)
        )
    results = await asyncio.gather(*tasks)

    # Parse each result
    outputs = [response_to_output(raw) for raw in results]

    # Majority vote on stage
    stage_counts = Counter(o.stage for o in outputs)
    majority_stage = stage_counts.most_common(1)[0][0]

    # If tied, prefer the earlier stage (conservative)
    if len(stage_counts) == ENSEMBLE_SIZE:  # all different
        # Pick the earliest in the developmental order
        for s in STAGES:
            if s in stage_counts:
                majority_stage = s
                break

    # Average confidence, combine reasoning
    avg_conf = sum(o.confidence for o in outputs) / len(outputs)
    votes = ", ".join(f"{o.stage}({o.confidence:.0%})" for o in outputs)
    majority_output = outputs[0]  # use first for reasoning base

    return PerceptionOutput(
        stage=majority_stage,
        confidence=avg_conf,
        reasoning=f"Ensemble [{votes}] → {majority_stage}. {majority_output.reasoning}",
        phase_count=ENSEMBLE_SIZE,
    )
