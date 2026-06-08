"""classify_stage — the submit tool. Never executed; terminates the agent loop."""
from __future__ import annotations

from harness.core.types import Stage

# `reasoning` MUST come before `stage`: with a forced tool call there's no
# free-text scratchpad, so the model deliberates inside the tool input. Putting
# `stage` first commits to an answer with no chain-of-thought, which the parity
# run showed pushes systematic over-advancement (pretzel→hatching, 2fold→pretzel).
CLASSIFY_TOOL_SCHEMA = {
    "name": "classify_stage",
    "description": (
        "Submit your final classification. Reason carefully about which reference "
        "images and stage descriptions match before committing to a stage."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Step-by-step analysis: which features you observe, which "
                    "reference images match, and why you ruled out adjacent stages."
                ),
            },
            "stage": {"type": "string", "enum": [s.value for s in Stage]},
        },
        "required": ["reasoning", "stage"],
        "additionalProperties": False,
    },
}
