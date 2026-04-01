"""
Setup script for Misviz data — downloads and verifies all required data.

Usage:
    python -m data_tools.misviz.setup_data --check        # Check what's available
    python -m data_tools.misviz.setup_data --clone-repo    # Clone GitHub repo
    python -m data_tools.misviz.setup_data --download-hf   # Download from HuggingFace
    python -m data_tools.misviz.setup_data --stats         # Print dataset statistics
"""
import argparse
import json
import subprocess
from pathlib import Path

from .config import (
    DATA_ROOT, MISVIZ_DIR, MISVIZ_SYNTH_DIR,
    MISVIZ_JSON, MISVIZ_SYNTH_JSON,
    MISVIZ_IMAGES_DIR, MISVIZ_SYNTH_IMAGES_DIR,
    MISVIZ_SYNTH_TABLES_DIR, MISVIZ_SYNTH_AXIS_DIR,
    TUDATALIB_URL, GITHUB_REPO,
)


def check_data_status():
    """Check what Misviz data is available locally."""
    print("=" * 60)
    print("  Misviz Data Status")
    print("=" * 60)

    checks = [
        ("Misviz JSON (real)", MISVIZ_JSON),
        ("Misviz-synth JSON", MISVIZ_SYNTH_JSON),
        ("Misviz images dir", MISVIZ_IMAGES_DIR),
        ("Misviz-synth images dir", MISVIZ_SYNTH_IMAGES_DIR),
        ("Misviz-synth data tables", MISVIZ_SYNTH_TABLES_DIR),
        ("Misviz-synth axis metadata", MISVIZ_SYNTH_AXIS_DIR),
    ]

    for label, path in checks:
        if path.exists():
            if path.is_file():
                size = path.stat().st_size / 1024
                print(f"  [OK] {label}: {path} ({size:.0f} KB)")
            else:
                count = sum(1 for _ in path.iterdir()) if path.is_dir() else 0
                print(f"  [OK] {label}: {path} ({count} files)")
        else:
            print(f"  [MISSING] {label}: {path}")

    print()
    print("  Next steps:")
    if not MISVIZ_JSON.exists() or not MISVIZ_SYNTH_JSON.exists():
        print(f"  1. Clone repo: git clone {GITHUB_REPO}")
        print(f"     Then copy data/ contents to {DATA_ROOT}/")
    if not MISVIZ_SYNTH_TABLES_DIR.exists():
        print(f"  2. Download synth data from TUdatalib: {TUDATALIB_URL}")
        print(f"     Extract to {MISVIZ_SYNTH_DIR}/")
    if not MISVIZ_IMAGES_DIR.exists():
        print(f"  3. Run image download script from the cloned repo:")
        print(f"     python data/download_misviz_images.py --use_wayback 0")


def clone_repo():
    """Clone the Misviz GitHub repo and copy data files."""
    repo_dir = DATA_ROOT / "arxiv2025-misviz"

    if repo_dir.exists():
        print(f"Repo already exists at {repo_dir}")
    else:
        print(f"Cloning {GITHUB_REPO}...")
        subprocess.run(["git", "clone", GITHUB_REPO, str(repo_dir)], check=True)

    # Copy JSON metadata files
    src_misviz = repo_dir / "data" / "misviz" / "misviz.json"
    src_synth = repo_dir / "data" / "misviz_synth" / "misviz_synth.json"

    if src_misviz.exists():
        MISVIZ_DIR.mkdir(parents=True, exist_ok=True)
        MISVIZ_JSON.write_bytes(src_misviz.read_bytes())
        print(f"  Copied misviz.json ({src_misviz.stat().st_size / 1024:.0f} KB)")

    if src_synth.exists():
        MISVIZ_SYNTH_DIR.mkdir(parents=True, exist_ok=True)
        MISVIZ_SYNTH_JSON.write_bytes(src_synth.read_bytes())
        print(f"  Copied misviz_synth.json ({src_synth.stat().st_size / 1024:.0f} KB)")

    # Copy download script
    dl_script = repo_dir / "data" / "download_misviz_images.py"
    if dl_script.exists():
        dest = MISVIZ_DIR / "download_images.py"
        dest.write_bytes(dl_script.read_bytes())
        print(f"  Copied download_images.py")

    print(f"\nDone. Data files in {DATA_ROOT}/")
    print(f"\nFor TUdatalib data (synth tables + axis metadata):")
    print(f"  Download manually from: {TUDATALIB_URL}")
    print(f"  Extract to: {MISVIZ_SYNTH_DIR}/")


def print_stats():
    """Print dataset statistics."""
    from .loader import MisvizLoader

    loader = MisvizLoader()

    for name, load_fn in [("Misviz-synth", loader.load_synth),
                           ("Misviz (real)", loader.load_real)]:
        try:
            data = load_fn()
        except FileNotFoundError as e:
            print(f"\n{name}: {e}")
            continue

        stats = loader.get_stats(data)
        print(f"\n{'='*60}")
        print(f"  {name}: {stats['total']} instances")
        print(f"{'='*60}")
        print(f"  Misleading: {stats['misleading']} ({stats['misleading_ratio']:.1%})")
        print(f"  Clean: {stats['clean']}")
        print(f"\n  Per split:")
        for split, count in stats["per_split"].items():
            print(f"    {split}: {count}")
        print(f"\n  Per misleader type:")
        for mtype, count in sorted(stats["per_misleader_type"].items(),
                                     key=lambda x: -x[1]):
            if count > 0:
                print(f"    {mtype}: {count}")
        print(f"\n  Per chart type:")
        for ctype, count in sorted(stats["per_chart_type"].items(),
                                     key=lambda x: -x[1])[:10]:
            print(f"    {ctype}: {count}")


def try_load_hf():
    """Try loading from HuggingFace."""
    try:
        from datasets import load_dataset
        print("Loading from HuggingFace (requires login)...")
        ds = load_dataset("UKPLab/misviz")
        print(f"Loaded: {ds}")
        for split in ds:
            print(f"  {split}: {len(ds[split])} instances")
    except Exception as e:
        print(f"Failed: {e}")
        print("You may need to: pip install datasets && huggingface-cli login")


def main():
    parser = argparse.ArgumentParser(description="Setup Misviz data")
    parser.add_argument("--check", action="store_true", help="Check data availability")
    parser.add_argument("--clone-repo", action="store_true", help="Clone GitHub repo")
    parser.add_argument("--download-hf", action="store_true", help="Try loading from HuggingFace")
    parser.add_argument("--stats", action="store_true", help="Print dataset statistics")
    args = parser.parse_args()

    if args.clone_repo:
        clone_repo()
    elif args.download_hf:
        try_load_hf()
    elif args.stats:
        print_stats()
    else:
        check_data_status()


if __name__ == "__main__":
    main()
