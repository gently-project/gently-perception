"""classify_stage — the submit tool. Never executed; terminates the agent loop."""
from __future__ import annotations

from harness.core.types import Stage

CLASSIFY_TOOL_SCHEMA = {
    "name": "classify_stage",
    "description": (
        "Submit your final classification. Call this when you have enough "
        "information to decide the developmental stage."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": [s.value for s in Stage]},
            "reasoning": {"type": "string", "description": "Brief justification."},
        },
        "required": ["stage", "reasoning"],
        "additionalProperties": False,
    },
}
