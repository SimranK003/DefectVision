import pytest
from defectvision.data.records import ImageRecord
from defectvision.data.split import stratified_split


def _make_records(cls: str, n: int) -> list[ImageRecord]:
    return [
        ImageRecord(
            basename=f"{cls}_{i}",
            image_path=None,
            annotation_path=None,
            folder_class=cls,
        )
        for i in range(n)
    ]


def test_split_ratios_are_respected_per_class():
    records = _make_records("crazing", 100) + _make_records("inclusion", 100)

    splits = stratified_split(records, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(splits["train"]) == 140
    assert len(splits["val"]) == 30
    assert len(splits["test"]) == 30

    # Per-class stratification: each class contributes proportionally, not
    # just the pooled total.
    train_classes = {r.folder_class for r in splits["train"]}
    assert train_classes == {"crazing", "inclusion"}
    crazing_in_train = sum(1 for r in splits["train"] if r.folder_class == "crazing")
    assert crazing_in_train == 70


def test_split_is_deterministic_given_same_seed():
    records = _make_records("crazing", 50)

    splits_a = stratified_split(records, 0.7, 0.15, 0.15, seed=7)
    splits_b = stratified_split(records, 0.7, 0.15, 0.15, seed=7)

    assert [r.basename for r in splits_a["train"]] == [r.basename for r in splits_b["train"]]


def test_no_overlap_between_splits():
    records = _make_records("crazing", 37)  # deliberately not evenly divisible

    splits = stratified_split(records, 0.7, 0.15, 0.15, seed=1)

    train_names = {r.basename for r in splits["train"]}
    val_names = {r.basename for r in splits["val"]}
    test_names = {r.basename for r in splits["test"]}

    assert train_names.isdisjoint(val_names)
    assert train_names.isdisjoint(test_names)
    assert val_names.isdisjoint(test_names)
    assert len(train_names) + len(val_names) + len(test_names) == 37


def test_invalid_ratios_raise():
    with pytest.raises(ValueError):
        stratified_split(_make_records("crazing", 10), 0.5, 0.3, 0.3, seed=1)
