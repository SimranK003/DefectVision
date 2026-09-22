#!/usr/bin/env python
"""Train the YOLOv8 defect detector from ml/configs/train.yaml.

Usage:
    python ml/scripts/train.py [--config ml/configs/train.yaml]

Registers the resulting weights + validation metrics in
ml/models/registry.json. Does NOT mark the run as production - that's a
separate, deliberate step (see ml/scripts/promote_model.py) so a training
run that regresses never silently becomes the model the API serves.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "ml" / "src"))

from defectvision.training.registry import add_entry  # noqa: E402


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_hardware(device: str) -> str:
    if device == "cuda":
        return f"GPU: {torch.cuda.get_device_name(0)} (CUDA)"
    if device == "mps":
        return f"GPU: Apple Silicon ({platform.processor() or platform.machine()}) via Metal/MPS"
    return f"CPU: {platform.processor() or platform.machine()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "ml/configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs (e.g. for smoke tests)")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Resume an interrupted run from a last.pt checkpoint (e.g. after a crash/OOM kill). "
            "Continues the same run directory, epoch count, and LR schedule."
        ),
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=["cpu", "mps", "cuda"],
        help="Override configs' device (e.g. move a resumed run to CPU after an MPS OOM).",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_yaml = REPO_ROOT / config["data_yaml"]
    if not data_yaml.exists():
        print(
            f"ERROR: {data_yaml} not found. Run `python ml/scripts/prepare_data.py` first.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(data_yaml) as f:
        dataset_version = yaml.safe_load(f)["dataset_version"]

    device = args.device or resolve_device(config["device"])
    hardware = describe_hardware(device)
    epochs = args.epochs or config["epochs"]

    print(f"[train] device={device} ({hardware})")
    print(f"[train] dataset_version={dataset_version}, epochs={epochs}")

    opt = config["optimizer"]
    aug = config["augmentation"]
    runs_dir = REPO_ROOT / config["runs_dir"]

    if args.resume:
        print(f"[train] resuming from {args.resume}")
        model = YOLO(str(args.resume))
        results = model.train(resume=True, device=device)
    else:
        model = YOLO(config["model"]["arch"])
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=config["image_size"],
            batch=config["batch_size"],
            patience=config["patience"],
            seed=config["seed"],
            device=device,
            optimizer=opt["name"],
            lr0=opt["lr0"],
            lrf=opt["lrf"],
            momentum=opt["momentum"],
            weight_decay=opt["weight_decay"],
            warmup_epochs=opt["warmup_epochs"],
            hsv_h=aug["hsv_h"],
            hsv_s=aug["hsv_s"],
            hsv_v=aug["hsv_v"],
            degrees=aug["degrees"],
            translate=aug["translate"],
            scale=aug["scale"],
            fliplr=aug["fliplr"],
            flipud=aug["flipud"],
            mosaic=aug["mosaic"],
            mixup=aug["mixup"],
            project=str(runs_dir),
            name=config["project_name"],
            exist_ok=False,
            plots=True,
            verbose=True,
        )

    run_dir = Path(results.save_dir)
    best_weights = run_dir / "weights" / "best.pt"

    # Validation metrics (the split used for early stopping / model
    # selection - NOT the held-out test set, see evaluate.py for that).
    # Routed to a throwaway subdir + plots disabled so this quick re-check
    # doesn't litter ml/runs/ with a second set of curve images.
    val_scratch_dir = runs_dir / "_val_scratch"
    val_metrics_obj = model.val(
        data=str(data_yaml),
        split="val",
        device=device,
        project=str(val_scratch_dir),
        name="check",
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=False,
    )
    shutil.rmtree(val_scratch_dir, ignore_errors=True)
    val_metrics = {
        "mAP50": float(val_metrics_obj.box.map50),
        "mAP50_95": float(val_metrics_obj.box.map),
        "precision": float(val_metrics_obj.box.mp),
        "recall": float(val_metrics_obj.box.mr),
    }
    print(f"[train] val metrics: {val_metrics}")

    with open(data_yaml) as f:
        classes = list(yaml.safe_load(f)["names"].values())

    registry_path = REPO_ROOT / "ml/models/registry.json"
    entry = add_entry(
        registry_path=registry_path,
        dataset_version=dataset_version,
        config={
            "train_config_path": str(args.config.relative_to(REPO_ROOT)),
            "epochs_run": epochs,
            "classes": classes,
        },
        run_dir=str(run_dir.relative_to(REPO_ROOT)),
        weights_path=str(best_weights.relative_to(REPO_ROOT)),
        val_metrics=val_metrics,
        hardware=hardware,
    )
    print(f"[train] registered as {entry['version']} in {registry_path}")


if __name__ == "__main__":
    main()
