"""Write a validated + split record set out as an Ultralytics-YOLO dataset.

Produces:

    processed_dir/
        images/{train,val,test}/<basename>.jpg   (copied, not symlinked -
                                                    keeps the dataset dir
                                                    self-contained/portable)
        labels/{train,val,test}/<basename>.txt    (YOLO normalized xywh)
        data.yaml                                  (Ultralytics dataset spec)
        splits.json                                (which basenames -> which split, for audit)

VOC pixel corners (xmin, ymin, xmax, ymax) are converted to YOLO's
normalized (x_center, y_center, width, height) in [0, 1], one line per box:
``<class_id> <x_center> <y_center> <width> <height>``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import yaml

from defectvision.data.records import ImageRecord


def _voc_box_to_yolo_line(box, class_to_id: dict[str, int], img_w: int, img_h: int) -> str:
    x_center = (box.xmin + box.xmax) / 2.0 / img_w
    y_center = (box.ymin + box.ymax) / 2.0 / img_h
    width = (box.xmax - box.xmin) / img_w
    height = (box.ymax - box.ymin) / img_h
    class_id = class_to_id[box.class_name]
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def write_yolo_dataset(
    splits: dict[str, list[ImageRecord]],
    classes: list[str],
    processed_dir: Path,
    dataset_version: str,
) -> Path:
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    class_to_id = {name: i for i, name in enumerate(classes)}
    splits_manifest: dict[str, list[str]] = {}

    for split_name, records in splits.items():
        img_out = processed_dir / "images" / split_name
        lbl_out = processed_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        basenames = []
        for record in records:
            width = record.actual_width
            height = record.actual_height
            dst_img = img_out / f"{record.basename}.jpg"
            shutil.copyfile(record.image_path, dst_img)

            lines = [
                _voc_box_to_yolo_line(box, class_to_id, width, height) for box in record.boxes
            ]
            (lbl_out / f"{record.basename}.txt").write_text("\n".join(lines) + "\n")
            basenames.append(record.basename)

        splits_manifest[split_name] = sorted(basenames)

    data_yaml = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
        "dataset_version": dataset_version,
    }
    (processed_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    (processed_dir / "splits.json").write_text(json.dumps(splits_manifest, indent=2))

    return processed_dir / "data.yaml"
