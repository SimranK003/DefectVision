"""Stratified, seeded train/val/test split over validated ImageRecords.

Stratifies by ``folder_class`` (every NEU-DET image contains defects of a
single class) so each split preserves the original 1:1:1:1:1:1 class
balance. Splitting is deterministic given the same seed and ratios, which
is what makes ``dataset_version`` in dataset.yaml meaningful.
"""

from __future__ import annotations

import random
from collections import defaultdict

from defectvision.data.records import ImageRecord


def stratified_split(
    records: list[ImageRecord],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[ImageRecord]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")

    by_class: dict[str, list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_class[r.folder_class].append(r)

    rng = random.Random(seed)
    splits: dict[str, list[ImageRecord]] = {"train": [], "val": [], "test": []}

    for cls, cls_records in sorted(by_class.items()):
        shuffled = cls_records[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = round(n * train_ratio)
        n_val = round(n * val_ratio)
        # remainder goes to test so counts always sum to n exactly
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train : n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val :])

    return splits
