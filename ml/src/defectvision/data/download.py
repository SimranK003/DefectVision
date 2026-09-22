"""Reproducible dataset acquisition for NEU-DET.

Clones the pinned mirror commit referenced in ``ml/configs/dataset.yaml`` so
that ``paths.raw_dir`` always ends up byte-identical across machines. Never
overwrites an existing raw_dir unless --force is passed, so re-running the
data pipeline doesn't silently re-download.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download_dataset(config: dict, force: bool = False) -> Path:
    raw_dir = REPO_ROOT / config["paths"]["raw_dir"]
    source = config["source"]

    if raw_dir.exists():
        if not force:
            print(f"[download] {raw_dir} already exists, skipping (use --force to re-download).")
            return raw_dir
        shutil.rmtree(raw_dir)

    raw_dir.parent.mkdir(parents=True, exist_ok=True)

    clone_dir = raw_dir.parent / "_mirror_clone_tmp"
    if clone_dir.exists():
        shutil.rmtree(clone_dir)

    print(f"[download] cloning {source['git_url']} @ {source['git_commit']}")
    subprocess.run(["git", "clone", "--quiet", source["git_url"], str(clone_dir)], check=True)
    subprocess.run(
        ["git", "-C", str(clone_dir), "checkout", "--quiet", source["git_commit"]],
        check=True,
    )

    src_subdir = clone_dir / source["subdir"]
    if not src_subdir.exists():
        raise FileNotFoundError(
            f"Expected subdir '{source['subdir']}' not found in cloned mirror at {clone_dir}"
        )

    shutil.copytree(src_subdir, raw_dir)
    shutil.rmtree(clone_dir)

    n_images = sum(1 for _ in raw_dir.rglob("*.jpg"))
    n_xml = sum(1 for _ in raw_dir.rglob("*.xml"))
    print(f"[download] done: {raw_dir} ({n_images} images, {n_xml} annotations)")
    return raw_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "ml/configs/dataset.yaml"
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if raw_dir exists")
    args = parser.parse_args()

    config = load_config(args.config)
    try:
        download_dataset(config, force=args.force)
    except subprocess.CalledProcessError as exc:
        print(f"[download] ERROR: git command failed ({exc}).", file=sys.stderr)
        print(
            "[download] The mirror may be unavailable. See README.md 'Dataset' "
            "section for manual download instructions (official NEU homepage / "
            "Kaggle mirrors).",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
