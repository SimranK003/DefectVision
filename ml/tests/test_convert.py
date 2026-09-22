from pathlib import Path

import yaml
from defectvision.data.convert import write_yolo_dataset
from defectvision.data.split import stratified_split
from defectvision.data.validate import validate_dataset
from fixtures import build_clean_dataset

CLASSES = ["crazing", "inclusion"]


def test_yolo_conversion_produces_normalized_boxes(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    build_clean_dataset(raw_dir, CLASSES, per_class=4)

    records, _ = validate_dataset(raw_dir, valid_classes=CLASSES)
    splits = stratified_split(records, 0.5, 0.25, 0.25, seed=0)

    data_yaml_path = write_yolo_dataset(splits, CLASSES, processed_dir, dataset_version="test-1")

    assert data_yaml_path.exists()
    spec = yaml.safe_load(data_yaml_path.read_text())
    assert spec["names"] == {0: "crazing", 1: "inclusion"}
    assert spec["dataset_version"] == "test-1"

    # Every image copied into a split must have a matching label file with
    # normalized (0-1) coordinates.
    for split in ("train", "val", "test"):
        img_dir = processed_dir / "images" / split
        lbl_dir = processed_dir / "labels" / split
        images = sorted(img_dir.glob("*.jpg"))
        assert images, f"expected at least one image in {split}"
        for img in images:
            label_path = lbl_dir / f"{img.stem}.txt"
            assert label_path.exists()
            class_id, xc, yc, w, h = label_path.read_text().strip().split()
            for v in (xc, yc, w, h):
                assert 0.0 <= float(v) <= 1.0
            assert int(class_id) in (0, 1)


def test_yolo_bbox_math_is_correct(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    build_clean_dataset(raw_dir, CLASSES, per_class=1)

    records, _ = validate_dataset(raw_dir, valid_classes=CLASSES)
    splits = {"train": records, "val": [], "test": []}
    write_yolo_dataset(splits, CLASSES, processed_dir, dataset_version="test-2")

    # fixtures.py always writes bbox (10,10,100,100) on a 200x200 image:
    # x_center=(10+100)/2/200=0.275, y_center=0.275, w=90/200=0.45, h=0.45
    label_file = next((processed_dir / "labels" / "train").glob("*.txt"))
    _, xc, yc, w, h = label_file.read_text().strip().split()
    assert abs(float(xc) - 0.275) < 1e-6
    assert abs(float(yc) - 0.275) < 1e-6
    assert abs(float(w) - 0.45) < 1e-6
    assert abs(float(h) - 0.45) < 1e-6
