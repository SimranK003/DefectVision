"""Pool + validate the raw NEU-DET mirror.

Walks every ``images/<class>/*.jpg`` and ``annotations/*.xml`` file under the
raw dataset directory (across both the mirror's ``train/`` and
``validation/`` folders - we don't trust that split, see dataset.yaml),
matches them by basename, parses VOC XML, and flags:

  * images that fail to open / are truncated
  * images with no matching annotation file (missing labels)
  * annotation files with no matching image (orphaned labels)
  * bounding boxes that are degenerate or out of image bounds
  * a mismatch between the XML <size> tag and the actual decoded image size
  * a mismatch between the folder-implied class and the XML object class

The output is a list of ``ImageRecord`` (only those with zero issues are
used by convert.py / split.py downstream) plus a summary dict used to render
the dataset report.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from defectvision.data.records import BBox, ImageRecord


def _parse_voc_xml(xml_path: Path) -> tuple[int | None, int | None, list[BBox]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    width = int(size_elem.findtext("width")) if size_elem is not None else None
    height = int(size_elem.findtext("height")) if size_elem is not None else None

    boxes: list[BBox] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if name is None or bnd is None:
            continue
        boxes.append(
            BBox(
                class_name=name.strip(),
                xmin=int(float(bnd.findtext("xmin"))),
                ymin=int(float(bnd.findtext("ymin"))),
                xmax=int(float(bnd.findtext("xmax"))),
                ymax=int(float(bnd.findtext("ymax"))),
            )
        )
    return width, height, boxes


def _collect_files(raw_dir: Path) -> tuple[dict[str, tuple[Path, str]], dict[str, Path]]:
    """Return (basename -> (image_path, folder_class)), (basename -> xml_path)."""
    images: dict[str, tuple[Path, str]] = {}
    for img_path in raw_dir.rglob("images/*/*.jpg"):
        basename = img_path.stem
        folder_class = img_path.parent.name
        images[basename] = (img_path, folder_class)

    annotations: dict[str, Path] = {}
    for xml_path in raw_dir.rglob("annotations/*.xml"):
        annotations[xml_path.stem] = xml_path

    return images, annotations


def validate_dataset(
    raw_dir: Path,
    valid_classes: list[str],
    min_bbox_area_px: int = 4,
) -> tuple[list[ImageRecord], dict]:
    images, annotations = _collect_files(raw_dir)
    all_basenames = sorted(set(images) | set(annotations))

    records: list[ImageRecord] = []
    valid_class_set = set(valid_classes)

    for basename in all_basenames:
        img_entry = images.get(basename)
        xml_path = annotations.get(basename)
        image_path, folder_class = img_entry if img_entry else (None, None)

        record = ImageRecord(
            basename=basename,
            image_path=image_path,
            annotation_path=xml_path,
            folder_class=folder_class,
        )

        if image_path is None:
            record.issues.append("missing_image")
        if xml_path is None:
            record.issues.append("missing_annotation")

        if image_path is not None:
            try:
                with Image.open(image_path) as im:
                    im.verify()
                with Image.open(image_path) as im:
                    record.actual_width, record.actual_height = im.size
            except (UnidentifiedImageError, OSError):
                record.issues.append("corrupted_image")

        if xml_path is not None:
            try:
                declared_w, declared_h, boxes = _parse_voc_xml(xml_path)
                record.declared_width = declared_w
                record.declared_height = declared_h
                record.boxes = boxes
            except ET.ParseError:
                record.issues.append("malformed_xml")
                boxes = []

            if not record.boxes and "malformed_xml" not in record.issues:
                record.issues.append("no_objects_in_annotation")

            for box in record.boxes:
                if box.class_name not in valid_class_set:
                    record.issues.append(f"unknown_class:{box.class_name}")
                if box.xmin >= box.xmax or box.ymin >= box.ymax:
                    record.issues.append("degenerate_bbox")
                if record.actual_width and (
                    box.xmin < 0
                    or box.ymin < 0
                    or box.xmax > record.actual_width
                    or box.ymax > (record.actual_height or 0)
                ):
                    record.issues.append("bbox_out_of_bounds")
                if box.area < min_bbox_area_px:
                    record.issues.append("bbox_too_small")

            if (
                record.actual_width
                and record.declared_width
                and (record.actual_width != record.declared_width or record.actual_height != record.declared_height)
            ):
                record.issues.append("size_mismatch")

            if folder_class and record.boxes and folder_class not in {b.class_name for b in record.boxes}:
                record.issues.append("folder_class_mismatch")

        records.append(record)

    summary = _summarize(records, valid_classes)
    return records, summary


def _summarize(records: list[ImageRecord], valid_classes: list[str]) -> dict:
    valid_records = [r for r in records if r.is_valid]
    issue_counter: Counter[str] = Counter()
    for r in records:
        for issue in r.issues:
            key = issue.split(":")[0]
            issue_counter[key] += 1

    class_image_counts = Counter()
    class_box_counts = Counter()
    for r in valid_records:
        for cls in r.classes_present:
            class_image_counts[cls] += 1
        for box in r.boxes:
            class_box_counts[box.class_name] += 1

    boxes_per_image = [len(r.boxes) for r in valid_records]

    return {
        "total_files_seen": len(records),
        "valid_images": len(valid_records),
        "invalid_images": len(records) - len(valid_records),
        "issue_counts": dict(issue_counter),
        "class_image_counts": {c: class_image_counts.get(c, 0) for c in valid_classes},
        "class_box_counts": {c: class_box_counts.get(c, 0) for c in valid_classes},
        "total_boxes": sum(class_box_counts.values()),
        "boxes_per_image_min": min(boxes_per_image) if boxes_per_image else 0,
        "boxes_per_image_max": max(boxes_per_image) if boxes_per_image else 0,
        "boxes_per_image_mean": (sum(boxes_per_image) / len(boxes_per_image)) if boxes_per_image else 0.0,
        "invalid_examples": [
            {"basename": r.basename, "issues": r.issues} for r in records if not r.is_valid
        ],
    }
