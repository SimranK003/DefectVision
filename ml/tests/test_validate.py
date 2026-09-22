from pathlib import Path

from defectvision.data.validate import validate_dataset
from fixtures import build_clean_dataset, build_dirty_dataset

CLASSES = ["crazing", "inclusion"]


def test_clean_dataset_has_no_issues(tmp_path: Path):
    build_clean_dataset(tmp_path, CLASSES, per_class=3)

    records, summary = validate_dataset(tmp_path, valid_classes=CLASSES)

    assert summary["total_files_seen"] == len(CLASSES) * 3 * 2  # train + validation folders
    assert summary["invalid_images"] == 0
    assert summary["issue_counts"] == {}
    assert all(r.is_valid for r in records)


def test_dirty_dataset_flags_every_known_issue(tmp_path: Path):
    issues = build_dirty_dataset(tmp_path, CLASSES)

    records, summary = validate_dataset(tmp_path, valid_classes=CLASSES)
    records_by_name = {r.basename: r for r in records}

    for basename, expected_issue in issues.items():
        assert expected_issue in records_by_name[basename].issues, (
            f"{basename} expected to be flagged with {expected_issue}, "
            f"got {records_by_name[basename].issues}"
        )

    # The one deliberately clean record must survive.
    assert records_by_name["clean"].is_valid
    assert summary["valid_images"] == 1


def test_class_distribution_counts_only_valid_images(tmp_path: Path):
    build_clean_dataset(tmp_path, CLASSES, per_class=5)

    _, summary = validate_dataset(tmp_path, valid_classes=CLASSES)

    # 5 per class per split (train+validation) = 10 per class total.
    assert summary["class_image_counts"]["crazing"] == 10
    assert summary["class_image_counts"]["inclusion"] == 10
    assert summary["total_boxes"] == 20
