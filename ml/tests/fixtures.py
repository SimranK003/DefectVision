"""Builds tiny synthetic NEU-DET-shaped raw datasets for fast, deterministic tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

VOC_TEMPLATE = """<annotation>
    <folder>{folder}</folder>
    <filename>{basename}.jpg</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>1</depth>
    </size>
    <object>
        <name>{class_name}</name>
        <bndbox>
            <xmin>{xmin}</xmin>
            <ymin>{ymin}</ymin>
            <xmax>{xmax}</xmax>
            <ymax>{ymax}</ymax>
        </bndbox>
    </object>
</annotation>
"""


def _write_image(path: Path, size: tuple[int, int] = (200, 200)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=128).save(path)


def _write_xml(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(VOC_TEMPLATE.format(**kwargs))


def build_clean_dataset(root: Path, classes: list[str], per_class: int = 4) -> None:
    """A perfectly valid dataset: every image has a matching, well-formed annotation."""
    for split in ("train", "validation"):
        for cls in classes:
            for i in range(per_class):
                basename = f"{cls}_{split}_{i}"
                _write_image(root / split / "images" / cls / f"{basename}.jpg")
                _write_xml(
                    root / split / "annotations" / f"{basename}.xml",
                    folder=cls,
                    basename=basename,
                    width=200,
                    height=200,
                    class_name=cls,
                    xmin=10,
                    ymin=10,
                    xmax=100,
                    ymax=100,
                )


def build_dirty_dataset(root: Path, classes: list[str]) -> dict[str, str]:
    """A dataset with one instance of each kind of data-quality problem.

    Returns a dict describing which basename carries which deliberate issue,
    for tests to assert against.
    """
    cls = classes[0]
    issues = {}

    # 1. Missing annotation entirely.
    _write_image(root / "train" / "images" / cls / "missing_ann.jpg")
    issues["missing_ann"] = "missing_annotation"

    # 2. Orphaned annotation (no image).
    _write_xml(
        root / "train" / "annotations" / "orphan_ann.xml",
        folder=cls,
        basename="orphan_ann",
        width=200,
        height=200,
        class_name=cls,
        xmin=10,
        ymin=10,
        xmax=100,
        ymax=100,
    )
    issues["orphan_ann"] = "missing_image"

    # 3. Degenerate bbox (xmax <= xmin).
    _write_image(root / "train" / "images" / cls / "degenerate.jpg")
    _write_xml(
        root / "train" / "annotations" / "degenerate.xml",
        folder=cls,
        basename="degenerate",
        width=200,
        height=200,
        class_name=cls,
        xmin=100,
        ymin=10,
        xmax=50,
        ymax=100,
    )
    issues["degenerate"] = "degenerate_bbox"

    # 4. Bounding box out of image bounds.
    _write_image(root / "train" / "images" / cls / "oob.jpg")
    _write_xml(
        root / "train" / "annotations" / "oob.xml",
        folder=cls,
        basename="oob",
        width=200,
        height=200,
        class_name=cls,
        xmin=10,
        ymin=10,
        xmax=999,
        ymax=999,
    )
    issues["oob"] = "bbox_out_of_bounds"

    # 5. Corrupted image (not a real image file).
    corrupt_path = root / "train" / "images" / cls / "corrupt.jpg"
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"not a real jpeg")
    _write_xml(
        root / "train" / "annotations" / "corrupt.xml",
        folder=cls,
        basename="corrupt",
        width=200,
        height=200,
        class_name=cls,
        xmin=10,
        ymin=10,
        xmax=100,
        ymax=100,
    )
    issues["corrupt"] = "corrupted_image"

    # 6. One clean valid record too, so validation isn't trivially "reject everything".
    _write_image(root / "train" / "images" / cls / "clean.jpg")
    _write_xml(
        root / "train" / "annotations" / "clean.xml",
        folder=cls,
        basename="clean",
        width=200,
        height=200,
        class_name=cls,
        xmin=10,
        ymin=10,
        xmax=100,
        ymax=100,
    )

    return issues
