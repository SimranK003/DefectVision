"""Render the dataset validation/EDA summary as JSON + Markdown + a chart."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot_class_distribution(split_counts: dict[str, dict[str, int]], out_path: Path) -> None:
    classes = list(next(iter(split_counts.values())).keys())
    splits = list(split_counts.keys())

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = [0] * len(classes)
    colors = {"train": "#3b6ea5", "val": "#e0a458", "test": "#6a994e"}
    for split in splits:
        values = [split_counts[split][c] for c in classes]
        ax.bar(classes, values, bottom=bottom, label=split, color=colors.get(split))
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_ylabel("Image count")
    ax.set_title("NEU-DET class distribution by split")
    ax.legend(title="Split")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def generate_report(
    validation_summary: dict,
    split_class_counts: dict[str, dict[str, int]],
    dataset_version: str,
    reports_dir: Path,
) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()

    full_report = {
        "dataset_version": dataset_version,
        "generated_at": timestamp,
        "validation": validation_summary,
        "split_class_counts": split_class_counts,
    }
    (reports_dir / "dataset_report.json").write_text(json.dumps(full_report, indent=2))

    chart_path = reports_dir / "class_distribution.png"
    _plot_class_distribution(split_class_counts, chart_path)

    md_lines = [
        "# NEU-DET Dataset Report",
        "",
        f"- **Dataset version:** {dataset_version}",
        f"- **Generated:** {timestamp}",
        "",
        "## Validation summary",
        "",
        f"- Files discovered: {validation_summary['total_files_seen']}",
        f"- Valid images (used downstream): {validation_summary['valid_images']}",
        f"- Invalid/excluded images: {validation_summary['invalid_images']}",
        f"- Total bounding boxes (valid images): {validation_summary['total_boxes']}",
        f"- Boxes per image: min={validation_summary['boxes_per_image_min']}, "
        f"max={validation_summary['boxes_per_image_max']}, "
        f"mean={validation_summary['boxes_per_image_mean']:.2f}",
        "",
        "### Issues found",
        "",
        "| Issue | Count |",
        "|---|---|",
    ]
    for issue, count in sorted(validation_summary["issue_counts"].items(), key=lambda kv: -kv[1]):
        md_lines.append(f"| {issue} | {count} |")
    if not validation_summary["issue_counts"]:
        md_lines.append("| (none) | 0 |")

    if validation_summary["invalid_examples"]:
        md_lines += ["", "### Excluded files (first 20)", "", "| File | Issues |", "|---|---|"]
        for ex in validation_summary["invalid_examples"][:20]:
            md_lines.append(f"| {ex['basename']} | {', '.join(ex['issues'])} |")

    md_lines += [
        "",
        "## Class distribution (post-validation, post-split)",
        "",
        "| Class | Train | Val | Test | Total |",
        "|---|---|---|---|---|",
    ]
    classes = list(next(iter(split_class_counts.values())).keys())
    for cls in classes:
        row = [split_class_counts[s][cls] for s in ("train", "val", "test")]
        md_lines.append(f"| {cls} | {row[0]} | {row[1]} | {row[2]} | {sum(row)} |")

    md_lines += ["", "![Class distribution](class_distribution.png)", ""]

    (reports_dir / "dataset_report.md").write_text("\n".join(md_lines))
