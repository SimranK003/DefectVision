"""Shared data model for a single NEU-DET image + its parsed annotation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BBox:
    class_name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)


@dataclass
class ImageRecord:
    """One image + its bounding boxes, plus any validation issues found."""

    basename: str
    image_path: Path | None
    annotation_path: Path | None
    folder_class: str | None  # class implied by the images/<class>/ subdirectory
    declared_width: int | None = None
    declared_height: int | None = None
    actual_width: int | None = None
    actual_height: int | None = None
    boxes: list[BBox] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.issues) == 0

    @property
    def classes_present(self) -> set[str]:
        return {b.class_name for b in self.boxes}
