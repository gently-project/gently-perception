"""
Compare perception function.

Includes the previous timepoint's image alongside the current one,
asking the model to compare them and determine if morphology has changed.

Uses module-level state to cache the previous image between calls.
When no previous image is available (first timepoint), falls back to
the hybrid approach.

Strategy: include both images, ask "has the morphology changed since
the last timepoint? If not, keep the same stage."
"""

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
    STAGES,
)

from .hybrid import _get_system_prompt, TEMPORAL_SYSTEM, SCIENTIFIC_SYSTEM

# Module-level state: cache the previous image
_prev_image: str | None = None
_prev_timepoint: int | None = None


COMPARE_ADDENDUM = """

## PREVIOUS TIMEPOINT IMAGE

You are shown TWO images: the PREVIOUS timepoint and the CURRENT timepoint.
Compare them directly:
- If the morphology looks essentially THE SAME → classify as the same stage
- If you see a CLEAR change (more folding, denser fill, new parallel bands) → \
advance to the next stage
- Small differences in brightness or orientation are NOT stage changes
"""


async def perceive_compare(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Classification with previous timepoint image comparison."""
    global _prev_image, _prev_timepoint

    # Detect new embryo (timepoint reset or big jump)
    if _prev_timepoint is not None and timepoint <= _prev_timepoint:
        _prev_image = None
        _prev_timepoint = None

    # Determine expected stage and system prompt (hybrid logic)
    last_stage = "early"
    if history:
        last_stage = history[-1].get("stage", "early")

    system_prompt = _get_system_prompt(last_stage)

    # Add comparison instructions if we have a previous image
    if _prev_image is not None:
        system_prompt = system_prompt + COMPARE_ADDENDUM

    content = build_reference_content(references)
    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})
        content.append({
            "type": "text",
            "text": (
                f"The most recent observation was '{last_stage}'. "
                f"Remember: stages change slowly. The current stage is most likely "
                f"'{last_stage}' unless you see a clear morphological change. "
                f"When uncertain, prefer the earlier stage."
            ),
        })

    # Include previous image if available
    if _prev_image is not None:
        content.append({"type": "text", "text": f"\nPREVIOUS TIMEPOINT (T{_prev_timepoint}) — classified as '{last_stage}':"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _prev_image,
            },
        })
        content.append({"type": "text", "text": f"\nCURRENT TIMEPOINT (T{timepoint}) — classify this:"})

    content.append({
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": image_b64,
        },
    })

    # Analysis prompt
    if _prev_image is not None:
        content.append({
            "type": "text",
            "text": (
                "Compare the CURRENT image to the PREVIOUS image. "
                "Has the morphology changed significantly? "
                "If the images look similar, classify as the same stage. "
                "If there's a clear change, classify accordingly."
            ),
        })
    else:
        # No previous image — use hybrid analysis prompts
        if system_prompt.startswith(SCIENTIFIC_SYSTEM[:50]):
            content.append({
                "type": "text",
                "text": (
                    "Analyze: (1) How much of the eggshell is filled with signal? "
                    "(2) How many parallel body segments are visible? "
                    "(3) Which reference images match best? Then classify."
                ),
            })
        else:
            content.append({
                "type": "text",
                "text": "Compare this image to the reference images above. Which stage's references does it most closely match? Classify accordingly.",
            })

    raw = await call_claude(system=system_prompt, content=content)
    result = response_to_output(raw)

    # Cache current image for next call
    _prev_image = image_b64
    _prev_timepoint = timepoint

    return result
