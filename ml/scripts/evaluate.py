#!/usr/bin/env python
"""Evaluate a registered model on the held-out TEST split (never seen during
training or during the early-stopping/model-selection val() calls in
train.py) and record honest numbers.

Produces:
    ml/data/reports/evaluation/metrics.json         overall + per-class metrics
    ml/data/reports/evaluation/evaluation_report.md  human-readable summary
    ml/data/reports/evaluation/confusion_matrix.png
    ml/data/reports/evaluation/PR_curve.png / F1 / P / R curves
    ml/data/reports/evaluation/sample_predictions.jpg

...and writes the same metrics into ml/models/registry.json's test_metrics
field for the evaluated version.

Usage:
    python ml/scripts/evaluate.py --version v1
    python ml/scripts/evaluate.py   # defaults to the most recently trained version
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from defectvision.training.registry import load_registry, set_test_metrics  # noqa: E402


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=None, help="Registry version to evaluate (default: most recent)")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "ml/configs/train.yaml")
    args = parser.parse_args()

    registry_path = REPO_ROOT / "ml/models/registry.json"
    entries = load_registry(registry_path)
    if not entries:
        print("ERROR: no trained models registered. Run ml/scripts/train.py first.", file=sys.stderr)
        sys.exit(1)

    entry = entries[-1] if args.version is None else next((e for e in entries if e["version"] == args.version), None)
    if entry is None:
        print(f"ERROR: no registry entry with version {args.version}", file=sys.stderr)
        sys.exit(1)

    with open(args.config) as f:
        train_config = yaml.safe_load(f)
    data_yaml = REPO_ROOT / train_config["data_yaml"]
    device = resolve_device(train_config["device"])

    weights_path = REPO_ROOT / entry["weights_path"]
    print(f"[evaluate] version={entry['version']} weights={weights_path} device={device}")

    eval_dir = REPO_ROOT / "ml/data/reports/evaluation"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True)

    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        device=device,
        project=str(eval_dir.parent),
        name=eval_dir.name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    per_class = []
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        p, r, ap50, ap = metrics.box.class_result(i)
        per_class.append(
            {
                "class_name": model.names[int(class_idx)],
                "precision": float(p),
                "recall": float(r),
                "mAP50": float(ap50),
                "mAP50_95": float(ap),
            }
        )

    test_metrics = {
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": "test",
    }

    # Ultralytics names its own prediction-grid images val_batchN_pred.jpg;
    # promote the first one found to a stable, predictable filename the
    # frontend's Model Information page links to directly.
    sample_candidates = sorted(eval_dir.glob("val_batch*_pred.jpg"))
    if sample_candidates:
        shutil.copyfile(sample_candidates[0], eval_dir / "sample_predictions.jpg")

    (eval_dir / "metrics.json").write_text(__import__("json").dumps(test_metrics, indent=2))

    md_lines = [
        "# Model Evaluation Report",
        "",
        f"- **Model version:** {entry['version']}",
        f"- **Dataset version:** {entry['dataset_version']}",
        "- **Evaluated on:** held-out TEST split (never used for training or model selection)",
        f"- **Evaluated at:** {test_metrics['evaluated_at']}",
        "",
        "## Overall metrics",
        "",
        f"- mAP@50: **{test_metrics['mAP50']:.3f}**",
        f"- mAP@50-95: **{test_metrics['mAP50_95']:.3f}**",
        f"- Precision: **{test_metrics['precision']:.3f}**",
        f"- Recall: **{test_metrics['recall']:.3f}**",
        "",
        "## Per-class metrics",
        "",
        "| Class | Precision | Recall | mAP@50 | mAP@50-95 |",
        "|---|---|---|---|---|",
    ]
    for row in per_class:
        md_lines.append(
            f"| {row['class_name']} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['mAP50']:.3f} | {row['mAP50_95']:.3f} |"
        )
    md_lines += [
        "",
        "## Artifacts",
        "",
        "![Confusion Matrix](confusion_matrix.png)",
        "",
        "![Precision-Recall Curve](PR_curve.png)",
        "",
        "![Sample Predictions](sample_predictions.jpg)",
        "",
    ]
    (eval_dir / "evaluation_report.md").write_text("\n".join(md_lines))

    set_test_metrics(registry_path, entry["version"], test_metrics)
    print(f"[evaluate] wrote {eval_dir}/evaluation_report.md")
    print(f"[evaluate] test mAP50={test_metrics['mAP50']:.3f} mAP50-95={test_metrics['mAP50_95']:.3f}")


if __name__ == "__main__":
    main()
