"""A tiny JSON-file model registry: one entry per training run.

Not a database - this is a portfolio project, and a flat file that's easy to
read in a code review is more honest than standing up MLflow for six classes
of steel defects. Each entry records exactly what produced a given weights
file so results are traceable: dataset_version + config + git-independent
timestamp + metrics. Exactly one entry may be marked ``is_production``; the
backend's GET /model endpoint reads whichever one that is.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_registry(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2))


def next_version(entries: list[dict[str, Any]]) -> str:
    existing = [int(e["version"].lstrip("v")) for e in entries if e["version"].lstrip("v").isdigit()]
    return f"v{max(existing, default=0) + 1}"


def add_entry(
    registry_path: Path,
    dataset_version: str,
    config: dict,
    run_dir: str,
    weights_path: str,
    val_metrics: dict,
    hardware: str,
) -> dict[str, Any]:
    entries = load_registry(registry_path)
    entry = {
        "version": next_version(entries),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset_version,
        "config": config,
        "run_dir": run_dir,
        "weights_path": weights_path,
        "val_metrics": val_metrics,
        "test_metrics": None,  # filled in by evaluate.py
        "hardware": hardware,
        "is_production": False,
    }
    entries.append(entry)
    save_registry(registry_path, entries)
    return entry


def set_test_metrics(registry_path: Path, version: str, test_metrics: dict) -> None:
    entries = load_registry(registry_path)
    for e in entries:
        if e["version"] == version:
            e["test_metrics"] = test_metrics
    save_registry(registry_path, entries)


def set_production(registry_path: Path, version: str) -> None:
    entries = load_registry(registry_path)
    found = False
    for e in entries:
        e["is_production"] = e["version"] == version
        found = found or e["is_production"]
    if not found:
        raise ValueError(f"No registry entry with version {version}")
    save_registry(registry_path, entries)


def get_production(registry_path: Path) -> dict[str, Any] | None:
    entries = load_registry(registry_path)
    for e in entries:
        if e["is_production"]:
            return e
    return None
