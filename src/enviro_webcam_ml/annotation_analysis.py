from __future__ import annotations

import csv
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enviro_webcam_ml.annotation import task_labels
from enviro_webcam_ml.config import AppConfig


@dataclass(frozen=True)
class PairAgreement:
    annotator_a: str
    annotator_b: str
    overlap_count: int
    agreement_count: int
    observed_agreement: float | None
    cohen_kappa: float | None


def analyze_annotations(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    task_id: str,
) -> dict[str, Any]:
    labels = task_labels(config, task_id)
    label_set = set(labels)
    rows = annotation_rows(conn, task_id=task_id)

    total_by_annotator: Counter[str] = Counter()
    count_by_annotator_label: dict[str, Counter[str]] = defaultdict(Counter)
    count_by_label: Counter[str] = Counter()
    legacy_labels: Counter[str] = Counter()
    by_capture: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        annotator = row["annotator"] or ""
        label = row["label"]
        total_by_annotator[annotator] += 1
        count_by_annotator_label[annotator][label] += 1
        count_by_label[label] += 1
        by_capture[int(row["capture_id"])].append(row)
        if label not in label_set:
            legacy_labels[label] += 1

    pair_agreements = compute_pair_agreements(by_capture, labels)
    disagreements = disagreement_rows(by_capture)

    double_labeled_count = sum(
        1
        for annotations in by_capture.values()
        if len({row["annotator"] or "" for row in annotations}) >= 2
    )

    return {
        "task_id": task_id,
        "configured_labels": labels,
        "annotation_count": len(rows),
        "unique_capture_count": len(by_capture),
        "double_labeled_capture_count": double_labeled_count,
        "total_by_annotator": dict(sorted(total_by_annotator.items())),
        "count_by_label": dict(sorted(count_by_label.items())),
        "count_by_annotator_label": {
            annotator: dict(sorted(counter.items()))
            for annotator, counter in sorted(count_by_annotator_label.items())
        },
        "legacy_labels": dict(sorted(legacy_labels.items())),
        "pair_agreements": pair_agreements,
        "disagreements": disagreements,
    }


def annotation_rows(conn: sqlite3.Connection, *, task_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
          a.capture_id,
          a.task_id,
          COALESCE(a.annotator, '') AS annotator,
          a.label,
          a.confidence,
          a.notes,
          a.created_at_utc AS annotated_at_utc,
          c.camera_id,
          c.captured_at_utc,
          ia.path AS image_path
        FROM annotation a
        LEFT JOIN capture c ON c.id = a.capture_id
        LEFT JOIN image_asset ia ON ia.capture_id = a.capture_id
        WHERE a.task_id = ?
        ORDER BY c.captured_at_utc, a.capture_id, annotator
        """,
        (task_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def compute_pair_agreements(
    by_capture: dict[int, list[dict[str, Any]]],
    configured_labels: list[str],
) -> list[PairAgreement]:
    annotators = sorted(
        {
            row["annotator"] or ""
            for annotations in by_capture.values()
            for row in annotations
        }
    )
    pairs: list[PairAgreement] = []
    labels = configured_labels[:]
    seen_labels = set(labels)

    for i, annotator_a in enumerate(annotators):
        for annotator_b in annotators[i + 1 :]:
            paired_labels: list[tuple[str, str]] = []
            for annotations in by_capture.values():
                label_a = latest_label_for_annotator(annotations, annotator_a)
                label_b = latest_label_for_annotator(annotations, annotator_b)
                if label_a is None or label_b is None:
                    continue
                paired_labels.append((label_a, label_b))
                if label_a not in seen_labels:
                    labels.append(label_a)
                    seen_labels.add(label_a)
                if label_b not in seen_labels:
                    labels.append(label_b)
                    seen_labels.add(label_b)

            agreement_count = sum(1 for label_a, label_b in paired_labels if label_a == label_b)
            observed = agreement_count / len(paired_labels) if paired_labels else None
            kappa = cohen_kappa(paired_labels, labels) if paired_labels else None
            pairs.append(
                PairAgreement(
                    annotator_a=annotator_a,
                    annotator_b=annotator_b,
                    overlap_count=len(paired_labels),
                    agreement_count=agreement_count,
                    observed_agreement=observed,
                    cohen_kappa=kappa,
                )
            )
    return pairs


def latest_label_for_annotator(
    annotations: list[dict[str, Any]],
    annotator: str,
) -> str | None:
    matches = [row for row in annotations if (row["annotator"] or "") == annotator]
    if not matches:
        return None
    matches.sort(key=lambda row: row.get("annotated_at_utc") or "")
    return str(matches[-1]["label"])


def cohen_kappa(paired_labels: list[tuple[str, str]], labels: list[str]) -> float | None:
    if not paired_labels:
        return None
    n = len(paired_labels)
    observed = sum(1 for a, b in paired_labels if a == b) / n
    counts_a = Counter(a for a, _ in paired_labels)
    counts_b = Counter(b for _, b in paired_labels)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in labels)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def disagreement_rows(by_capture: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for capture_id, annotations in sorted(by_capture.items()):
        by_annotator = {
            row["annotator"] or "": row
            for row in sorted(annotations, key=lambda item: item.get("annotated_at_utc") or "")
        }
        labels = {row["label"] for row in by_annotator.values()}
        if len(by_annotator) < 2 or len(labels) <= 1:
            continue
        first = next(iter(by_annotator.values()))
        result.append(
            {
                "capture_id": capture_id,
                "captured_at_utc": first.get("captured_at_utc"),
                "camera_id": first.get("camera_id"),
                "image_path": first.get("image_path"),
                "annotations": {
                    annotator: row["label"]
                    for annotator, row in sorted(by_annotator.items())
                },
            }
        )
    return result


def write_analysis_markdown(analysis: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Annotation analysis: `{analysis['task_id']}`",
        "",
        f"- Total annotations: {analysis['annotation_count']}",
        f"- Unique captures annotated: {analysis['unique_capture_count']}",
        f"- Captures with 2+ annotators: {analysis['double_labeled_capture_count']}",
        "",
        "## Configured labels",
        "",
        *[f"- `{label}`" for label in analysis["configured_labels"]],
        "",
        "## Counts by annotator",
        "",
    ]
    if analysis["total_by_annotator"]:
        lines.extend(
            f"- `{annotator or 'unknown'}`: {count}"
            for annotator, count in analysis["total_by_annotator"].items()
        )
    else:
        lines.append("- No annotations found.")

    lines.extend(["", "## Counts by label", ""])
    lines.extend(
        f"- `{label}`: {count}"
        for label, count in analysis["count_by_label"].items()
    )
    if not analysis["count_by_label"]:
        lines.append("- No labels found.")

    lines.extend(["", "## Multi-rater agreement", ""])
    pair_agreements: list[PairAgreement] = analysis["pair_agreements"]
    if pair_agreements:
        lines.extend(
            [
                "| Annotator A | Annotator B | Overlap | Agreements | Agreement | Cohen's kappa |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for pair in pair_agreements:
            lines.append(
                "| "
                f"`{pair.annotator_a or 'unknown'}` | "
                f"`{pair.annotator_b or 'unknown'}` | "
                f"{pair.overlap_count} | "
                f"{pair.agreement_count} | "
                f"{format_percent(pair.observed_agreement)} | "
                f"{format_float(pair.cohen_kappa)} |"
            )
    else:
        lines.append("- Need at least two annotators for agreement statistics.")

    lines.extend(["", "## Legacy labels not in current config", ""])
    if analysis["legacy_labels"]:
        lines.append(
            "These labels exist in the database but are not part of the current label set. "
            "Do not automatically split ambiguous legacy labels; re-review those frames."
        )
        lines.append("")
        lines.extend(
            f"- `{label}`: {count}"
            for label, count in analysis["legacy_labels"].items()
        )
    else:
        lines.append("- None.")

    lines.extend(["", "## Disagreements", ""])
    disagreements = analysis["disagreements"]
    if disagreements:
        lines.append(f"- Disagreement count: {len(disagreements)}")
        lines.append("- See the disagreement CSV for image paths and per-annotator labels.")
    else:
        lines.append("- No disagreements found among overlapping annotations.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_disagreements_csv(disagreements: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotators = sorted(
        {
            annotator
            for row in disagreements
            for annotator in row["annotations"].keys()
        }
    )
    fieldnames = ["capture_id", "captured_at_utc", "camera_id", "image_path", *annotators]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in disagreements:
            output = {
                "capture_id": row["capture_id"],
                "captured_at_utc": row["captured_at_utc"],
                "camera_id": row["camera_id"],
                "image_path": row["image_path"],
            }
            output.update(row["annotations"])
            writer.writerow(output)


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"
