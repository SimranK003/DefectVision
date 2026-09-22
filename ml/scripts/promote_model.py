#!/usr/bin/env python
"""Mark one registry entry as the production model.

Promotion is a deliberate, separate step from training on purpose: a
training run that regresses (or is still mid-training) should never
silently become what the API serves. Run this only after reviewing
ml/scripts/evaluate.py's output for the version you're about to promote.

Usage:
    python ml/scripts/promote_model.py v2
    python ml/scripts/promote_model.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from defectvision.training.registry import load_registry, set_production  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="Version to promote, e.g. v2")
    parser.add_argument("--list", action="store_true", help="List all registered versions and exit")
    args = parser.parse_args()

    registry_path = REPO_ROOT / "ml/models/registry.json"

    if args.list or not args.version:
        entries = load_registry(registry_path)
        if not entries:
            print("No trained models registered yet.")
            return
        for e in entries:
            marker = " (PRODUCTION)" if e["is_production"] else ""
            test_map = (e.get("test_metrics") or {}).get("mAP50")
            val_map = e["val_metrics"]["mAP50"]
            map_str = f"test mAP50={test_map:.3f}" if test_map is not None else f"val mAP50={val_map:.3f}"
            print(f"{e['version']}{marker}: {e['created_at']} - {map_str} - {e['weights_path']}")
        if not args.version:
            return

    set_production(registry_path, args.version)
    print(f"[promote] {args.version} is now the production model.")


if __name__ == "__main__":
    main()
