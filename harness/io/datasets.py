"""
Dataset registry.

A dataset bundles a volume source location with its ground-truth file(s) and
the HuggingFace repo to pull from. Tasks reference a dataset by name; the
runner resolves it here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Dataset:
    name: str
    hf_repo: str
    hf_pattern: str
    volumes_dir: Path
    onset_gt: Path | None = None
    stage_gt: Path | None = None
    notes: str = ""
    embryo_filter: tuple[str, ...] = field(default_factory=tuple)
    """If non-empty, restrict to these embryo_ids by default."""


DATASETS: dict[str, Dataset] = {
    "stage": Dataset(
        name="stage",
        hf_repo="pskeshu/gently-perception-benchmark",
        hf_pattern="volumes/**",
        volumes_dir=DATA_DIR / "volumes",
        stage_gt=DATA_DIR / "ground_truth" / "59799c78.json",
    ),
    "sls762": Dataset(
        name="sls762",
        hf_repo="pskeshu/perception-benchmark",
        hf_pattern="celegans_dopaminergic_sls762/**",
        volumes_dir=DATA_DIR / "sls762" / "celegans_dopaminergic_sls762" / "volumes",
        onset_gt=DATA_DIR / "ground_truth" / "onset" / "sls762.json",
        notes="dat-1p::mNeonGreen dopaminergic reporter. Gated: run `hf auth login`.",
        embryo_filter=("embryo_003", "embryo_004", "embryo_005", "embryo_006"),
    ),
}
