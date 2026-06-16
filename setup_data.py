"""
Download benchmark data from HuggingFace.

Pulls the C. elegans light-sheet volume data from the HuggingFace dataset
repo and places it in data/volumes/ where the benchmark harness expects it.

Usage:
    python setup_data.py              # Download volumes
    python setup_data.py --dry-run    # Show what would be downloaded
    python setup_data.py --filmstrip  # Download + generate filmstrip viewer
"""

import argparse
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

DATASETS = {
    "stage": {
        "repo": "pskeshu/gently-perception-benchmark",
        "pattern": "volumes/**",
        "target": DATA_DIR / "volumes",
        "size": "~35 GB",
    },
    "sls762": {
        "repo": "pskeshu/perception-benchmark",
        "pattern": "celegans_dopaminergic_sls762/**",
        "target": DATA_DIR / "sls762",
        "size": "~1.5 GB",
    },
}

# Backward-compat for any external scripts that imported these.
HF_REPO = DATASETS["stage"]["repo"]
VOLUMES_DIR = DATASETS["stage"]["target"]


def download_data(dataset: str = "stage", dry_run: bool = False):
    """Download volume data from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    cfg = DATASETS[dataset]
    repo, pattern, target = cfg["repo"], cfg["pattern"], cfg["target"]
    print(f"Dataset: {repo}  (pattern: {pattern})")
    print(f"Target:  {target}")

    if target.exists() and any(target.iterdir()):
        n_embryos = len([d for d in target.iterdir() if d.is_dir()])
        print(f"\n{target.name}/ already exists with {n_embryos} embryo folders.")
        print("Re-running will only download new/changed files.")

    if dry_run:
        print("\n[DRY RUN] Would download from:")
        print(f"  https://huggingface.co/datasets/{repo}")
        print(f"  -> {target}/")
        print(f"\nRun without --dry-run to download ({cfg['size']}).")
        return

    print(f"\nDownloading ({cfg['size']})...\n")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=str(target),
        allow_patterns=[pattern],
    )

    if target.exists():
        files = list(target.rglob("*.tif")) + list(target.rglob("*.tiff")) + list(target.rglob("*.npz"))
        print(f"\nDone! {len(files)} volume files under {target}.")
    else:
        print(f"\nWarning: {target} not found after download.")
        print("Check the HuggingFace repo structure.")


def generate_filmstrip():
    """Generate the filmstrip HTML viewer."""
    print("\nGenerating filmstrip viewer...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "make_filmstrip.py"],
        cwd=Path(__file__).parent,
    )
    if result.returncode == 0:
        print("Filmstrip viewer: data/filmstrip/viewer.html")
    else:
        print("Filmstrip generation failed. You can retry with: python make_filmstrip.py")


def main():
    parser = argparse.ArgumentParser(
        description="Download benchmark data from HuggingFace"
    )
    parser.add_argument(
        "--dataset", choices=sorted(DATASETS), default="stage",
        help="Which dataset to download",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--filmstrip", action="store_true",
        help="Generate filmstrip HTML viewer after downloading",
    )
    args = parser.parse_args()

    download_data(dataset=args.dataset, dry_run=args.dry_run)

    if args.filmstrip and not args.dry_run:
        generate_filmstrip()


if __name__ == "__main__":
    main()
