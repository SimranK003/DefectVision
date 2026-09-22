#!/usr/bin/env python
"""End-to-end reproducible data pipeline: download -> validate -> split -> convert -> report.

Usage:
    python ml/scripts/prepare_data.py [--config ml/configs/dataset.yaml] [--force-download]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from defectvision.data.convert import write_yolo_dataset  # noqa: E402
from defectvision.data.download import download_dataset  # noqa: E402
from defectvision.data.report import generate_report  # noqa: E402
from defectvision.data.split import stratified_split  # noqa: E402
from defectvision.data.validate import validate_dataset  # noqa: E402


def _split_class_counts(splits, classes) -> dict[str, dict[str, int]]:
    out = {}
    for split_name, records in splits.items():
        counter = Counter()
        for r in records:
            counter[r.folder_class] += 1
        out[split_name] = {c: counter.get(c, 0) for c in classes}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "ml/configs/dataset.yaml")
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    raw_dir = REPO_ROOT / config["paths"]["raw_dir"]
    processed_dir = REPO_ROOT / config["paths"]["processed_dir"]
    reports_dir = REPO_ROOT / config["paths"]["reports_dir"]
    classes = config["classes"]

    print("== Step 1/4: download ==")
    download_dataset(config, force=args.force_download)

    print("== Step 2/4: validate ==")
    records, summary = validate_dataset(
        raw_dir=raw_dir,
        valid_classes=classes,
        min_bbox_area_px=config["validation"]["min_bbox_area_px"],
    )
    print(
        f"  {summary['valid_images']} valid / {summary['total_files_seen']} total "
        f"({summary['invalid_images']} excluded)"
    )
    if summary["issue_counts"]:
        print(f"  issues: {summary['issue_counts']}")

    valid_records = [r for r in records if r.is_valid]
    if not valid_records:
        print("ERROR: no valid records survived validation.", file=sys.stderr)
        sys.exit(1)

    print("== Step 3/4: split + convert to YOLO format ==")
    splits = stratified_split(
        valid_records,
        train_ratio=config["split"]["train"],
        val_ratio=config["split"]["val"],
        test_ratio=config["split"]["test"],
        seed=config["split"]["seed"],
    )
    for name, recs in splits.items():
        print(f"  {name}: {len(recs)} images")

    data_yaml_path = write_yolo_dataset(
        splits=splits,
        classes=classes,
        processed_dir=processed_dir,
        dataset_version=config["dataset_version"],
    )
    print(f"  wrote {data_yaml_path}")

    print("== Step 4/4: dataset report ==")
    split_counts = _split_class_counts(splits, classes)
    generate_report(
        validation_summary=summary,
        split_class_counts=split_counts,
        dataset_version=config["dataset_version"],
        reports_dir=reports_dir,
    )
    print(f"  wrote {reports_dir / 'dataset_report.md'}")


if __name__ == "__main__":
    main()
