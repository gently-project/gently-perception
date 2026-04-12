"""Reference image loading.

Loads stage-organized reference images from package data or a custom path.
These are the few-shot examples provided to the VLM for classification.
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from .api import STAGES

logger = logging.getLogger(__name__)

# Package-bundled examples directory
_PACKAGE_EXAMPLES_DIR = Path(__file__).parent.parent / "data" / "examples"


def load_examples(
    path: Optional[Path] = None,
    max_per_stage: int = 2,
) -> dict[str, list[str]]:
    """Load reference images for each stage.

    Parameters
    ----------
    path : Path, optional
        Directory containing stage subdirectories with images.
        Defaults to package-bundled examples at ``data/examples/``.
    max_per_stage : int
        Maximum number of example images per stage.

    Returns
    -------
    dict mapping stage name to list of base64-encoded JPEG strings.
    """
    examples_dir = path or _PACKAGE_EXAMPLES_DIR

    if not examples_dir.exists():
        logger.warning(f"Examples directory not found: {examples_dir}")
        return {}

    references: dict[str, list[str]] = {}

    for stage in STAGES:
        stage_dir = examples_dir / stage
        if not stage_dir.exists():
            continue

        images: list[str] = []
        for img_path in sorted(stage_dir.iterdir()):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                image_data = img_path.read_bytes()
                images.append(base64.b64encode(image_data).decode("utf-8"))
                if len(images) >= max_per_stage:
                    break

        if images:
            references[stage] = images

    loaded_count = sum(len(v) for v in references.values())
    logger.info(
        f"Loaded {loaded_count} reference images "
        f"for {len(references)} stages from {examples_dir}"
    )

    return references
