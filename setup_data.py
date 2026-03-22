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

HF_REPO = "pskeshu/gently-perception-benchmark"
DATA_DIR = Path(__file__).parent / "data"
VOLUMES_DIR = DATA_DIR / "volumes"


def download_data(dry_run=False):
    """Download volume data from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    print(f"Dataset: {HF_REPO}")
    print(f"Target:  {VOLUMES_DIR}")

    if VOLUMES_DIR.exists() and any(VOLUMES_DIR.iterdir()):
        n_embryos = len([d for d in VOLUMES_DIR.iterdir() if d.is_dir()])
        print(f"\ndata/volumes/ already exists with {n_embryos} embryo folders.")
        print("Re-running will only download new/changed files.")

    if dry_run:
        print("\n[DRY RUN] Would download from:")
        print(f"  https://huggingface.co/datasets/{HF_REPO}")
        print(f"  -> {VOLUMES_DIR}/")
        print("\nRun without --dry-run to download (~35 GB).")
        return

    print(f"\nDownloading volumes (~35 GB)...")
    print("This may take a while on the first run.\n")

    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(DATA_DIR),
        allow_patterns=["volumes/**"],
    )

    # Verify
    if VOLUMES_DIR.exists():
        embryos = [d for d in VOLUMES_DIR.iterdir() if d.is_dir()]
        total_files = sum(1 for e in embryos for _ in e.glob("*.tif"))
        print(f"\nDone! {len(embryos)} embryos, {total_files} volume files.")
    else:
        print("\nWarning: data/volumes/ not found after download.")
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
        "--dry-run", action="store_true",
        help="Show what would be downloaded without downloading",
    )
    parser.add_argument(
        "--filmstrip", action="store_true",
        help="Generate filmstrip HTML viewer after downloading",
    )
    args = parser.parse_args()

    download_data(dry_run=args.dry_run)

    if args.filmstrip and not args.dry_run:
        generate_filmstrip()


if __name__ == "__main__":
    main()
