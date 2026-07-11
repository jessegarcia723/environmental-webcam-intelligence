from __future__ import annotations

import csv
import html
import math
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from PIL import Image, ImageDraw

from enviro_webcam_ml.config import AppConfig
from enviro_webcam_ml.image_preprocessing import PixelCrop, crop_image, crop_to_dict
from enviro_webcam_ml.image_paths import resolve_image_path
from enviro_webcam_ml.training_dataset import (
    adjudication_labels,
    candidate_rows,
    group_latest_annotations,
)


POSITIVE_EVENT_LABEL = "both_cameras_clouds_below_peak"
NEGATIVE_EVENT_LABEL = "not_both_cameras_clouds_below_peak"


@dataclass(frozen=True)
class PairedEventOptions:
    task_id: str
    output_dir: Path
    camera_ids: tuple[str, str]
    positive_label: str
    min_annotators: int = 2
    max_pair_minutes: float = 3.0
    thumbnail_width: int = 640
    crop_pixels: PixelCrop | dict[str, int] | None = None
    timezone_name: str = "America/Los_Angeles"


def build_paired_events(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    options: PairedEventOptions,
) -> dict[str, Any]:
    if len(options.camera_ids) != 2:
        raise ValueError("Paired events require exactly two camera IDs.")
    if options.max_pair_minutes <= 0:
        raise ValueError("max_pair_minutes must be greater than 0.")
    if options.thumbnail_width <= 0:
        raise ValueError("thumbnail_width must be greater than 0.")

    labeled_frames, skipped_frames = labeled_frame_rows(conn, config=config, options=options)
    events, skipped_pairs = pair_labeled_frames(labeled_frames, options=options)

    options.output_dir.mkdir(parents=True, exist_ok=True)
    events_csv = options.output_dir / "paired_events.csv"
    summary_md = options.output_dir / "paired_event_summary.md"
    hour_csv = options.output_dir / "paired_event_hour_histogram.csv"
    hour_png = options.output_dir / "paired_event_hour_histogram.png"
    examples_dir = options.output_dir / "both_cameras_clouds_below_peak_examples"
    examples_html = examples_dir / "index.html"

    write_events_csv(events, events_csv)
    hour_rows = hourly_summary(events)
    write_hour_csv(hour_rows, hour_csv)
    plot_hour_histogram(hour_rows, hour_png)
    gallery = write_positive_event_gallery(
        events,
        examples_dir,
        crop_pixels=options.crop_pixels,
        thumbnail_width=options.thumbnail_width,
    )
    summary = summarize_events(
        events,
        skipped_frames=skipped_frames,
        skipped_pairs=skipped_pairs,
        options=options,
        paths={
            "events_csv": events_csv,
            "summary_md": summary_md,
            "hour_csv": hour_csv,
            "hour_png": hour_png,
            "examples_html": examples_html,
            "examples_dir": examples_dir,
        },
        gallery=gallery,
    )
    write_summary_markdown(summary, summary_md)
    return summary


def labeled_frame_rows(
    conn: sqlite3.Connection,
    *,
    config: AppConfig,
    options: PairedEventOptions,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows = candidate_rows(conn, task_id=options.task_id)
    grouped = group_latest_annotations(rows)
    adjudications = adjudication_labels(conn, task_id=options.task_id)
    selected = []
    skipped: Counter[str] = Counter()
    allowed_cameras = set(options.camera_ids)

    for capture_id, annotations in sorted(grouped.items()):
        annotators = sorted(annotations)
        if not annotators:
            skipped["no_annotations"] += 1
            continue
        labels = {annotations[annotator]["label"] for annotator in annotators}
        first = annotations[annotators[0]]
        if first.get("camera_id") not in allowed_cameras:
            skipped["other_camera"] += 1
            continue
        adjudicated_label = adjudications.get(capture_id)
        if len(annotators) < options.min_annotators and adjudicated_label is None:
            skipped["too_few_annotators"] += 1
            continue
        if adjudicated_label is not None:
            label = adjudicated_label
            label_source = "adjudicated"
        elif len(labels) == 1:
            label = next(iter(labels))
            label_source = "agreement"
        else:
            skipped["disagreement"] += 1
            continue

        image_path = resolve_image_path(first.get("image_path"), config.data_dir)
        image_exists = bool(image_path and image_path.exists())
        if not image_exists:
            skipped["missing_image"] += 1
            continue
        captured_at_utc = first.get("captured_at_utc") or ""
        if not captured_at_utc:
            skipped["missing_capture_time"] += 1
            continue
        selected.append(
            {
                "capture_id": int(capture_id),
                "camera_id": first.get("camera_id") or "",
                "captured_at_utc": captured_at_utc,
                "captured_at": parse_datetime(captured_at_utc),
                "label": label,
                "label_source": label_source,
                "image_path": str(image_path),
                "original_image_path": first.get("image_path") or "",
                "annotators": "|".join(annotators),
                "annotator_count": len(annotators),
            }
        )

    selected.sort(key=lambda row: (row["captured_at"], row["camera_id"], row["capture_id"]))
    return selected, dict(sorted(skipped.items()))


def pair_labeled_frames(
    frames: list[dict[str, Any]],
    *,
    options: PairedEventOptions,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    camera_a, camera_b = options.camera_ids
    rows_a = [row for row in frames if row["camera_id"] == camera_a]
    rows_b = [row for row in frames if row["camera_id"] == camera_b]
    unused_b = set(range(len(rows_b)))
    events = []
    skipped: Counter[str] = Counter()
    max_delta_seconds = options.max_pair_minutes * 60.0
    local_tz = ZoneInfo(options.timezone_name)

    for row_a in rows_a:
        best_index = None
        best_delta = None
        for index in sorted(unused_b):
            row_b = rows_b[index]
            delta = abs((row_a["captured_at"] - row_b["captured_at"]).total_seconds())
            if delta <= max_delta_seconds and (best_delta is None or delta < best_delta):
                best_index = index
                best_delta = delta
        if best_index is None:
            skipped[f"unpaired_{camera_a}"] += 1
            continue
        unused_b.remove(best_index)
        row_b = rows_b[best_index]
        midpoint = row_a["captured_at"] + (row_b["captured_at"] - row_a["captured_at"]) / 2
        local_time = midpoint.astimezone(local_tz)
        event_label = (
            POSITIVE_EVENT_LABEL
            if row_a["label"] == options.positive_label and row_b["label"] == options.positive_label
            else NEGATIVE_EVENT_LABEL
        )
        events.append(
            {
                "event_id": event_id(midpoint, row_a, row_b),
                "event_time_utc": midpoint.isoformat(),
                "event_time_local": local_time.isoformat(),
                "local_date": local_time.date().isoformat(),
                "local_hour": local_time.hour,
                "local_decimal_hour": local_time.hour + local_time.minute / 60 + local_time.second / 3600,
                "event_label": event_label,
                "is_both_positive": int(event_label == POSITIVE_EVENT_LABEL),
                "pair_delta_seconds": round(float(best_delta or 0.0), 3),
                f"{camera_a}_capture_id": row_a["capture_id"],
                f"{camera_a}_captured_at_utc": row_a["captured_at_utc"],
                f"{camera_a}_label": row_a["label"],
                f"{camera_a}_label_source": row_a["label_source"],
                f"{camera_a}_image_path": row_a["image_path"],
                f"{camera_b}_capture_id": row_b["capture_id"],
                f"{camera_b}_captured_at_utc": row_b["captured_at_utc"],
                f"{camera_b}_label": row_b["label"],
                f"{camera_b}_label_source": row_b["label_source"],
                f"{camera_b}_image_path": row_b["image_path"],
                "label_pair": f"{row_a['label']}|{row_b['label']}",
            }
        )

    skipped[f"unpaired_{camera_b}"] += len(unused_b)
    events.sort(key=lambda row: (row["event_time_utc"], row["event_id"]))
    return events, dict(sorted(skipped.items()))


def event_id(midpoint: datetime, row_a: dict[str, Any], row_b: dict[str, Any]) -> str:
    stamp = midpoint.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{row_a['capture_id']}_{row_b['capture_id']}"


def hourly_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hour: dict[int, Counter[str]] = {hour: Counter() for hour in range(24)}
    for event in events:
        hour = int(event["local_hour"])
        by_hour[hour]["event_count"] += 1
        if event["event_label"] == POSITIVE_EVENT_LABEL:
            by_hour[hour]["both_positive_count"] += 1
        else:
            by_hour[hour]["not_both_positive_count"] += 1
    rows = []
    for hour in range(24):
        count = by_hour[hour]["event_count"]
        positive = by_hour[hour]["both_positive_count"]
        rows.append(
            {
                "local_hour": hour,
                "event_count": count,
                "both_positive_count": positive,
                "not_both_positive_count": by_hour[hour]["not_both_positive_count"],
                "both_positive_rate": positive / count if count else "",
            }
        )
    return rows


def summarize_events(
    events: list[dict[str, Any]],
    *,
    skipped_frames: dict[str, int],
    skipped_pairs: dict[str, int],
    options: PairedEventOptions,
    paths: dict[str, Path],
    gallery: dict[str, Any],
) -> dict[str, Any]:
    label_counts = Counter(row["event_label"] for row in events)
    pair_counts = Counter(row["label_pair"] for row in events)
    positive_events = [row for row in events if row["event_label"] == POSITIVE_EVENT_LABEL]
    positive_by_hour = Counter(int(row["local_hour"]) for row in positive_events)
    top_positive_hours = [
        {"local_hour": hour, "both_positive_count": count}
        for hour, count in positive_by_hour.most_common()
    ]
    return {
        "task_id": options.task_id,
        "camera_ids": list(options.camera_ids),
        "positive_label": options.positive_label,
        "event_positive_label": POSITIVE_EVENT_LABEL,
        "max_pair_minutes": options.max_pair_minutes,
        "timezone": options.timezone_name,
        "crop_pixels": crop_to_dict(options.crop_pixels),
        "event_count": len(events),
        "both_positive_count": label_counts.get(POSITIVE_EVENT_LABEL, 0),
        "not_both_positive_count": label_counts.get(NEGATIVE_EVENT_LABEL, 0),
        "both_positive_rate": safe_divide(label_counts.get(POSITIVE_EVENT_LABEL, 0), len(events)),
        "event_label_counts": dict(sorted(label_counts.items())),
        "single_label_pair_counts": dict(sorted(pair_counts.items())),
        "top_positive_hours": top_positive_hours,
        "skipped_frames": skipped_frames,
        "skipped_pairs": skipped_pairs,
        "paths": {key: str(path) for key, path in paths.items()},
        "gallery": gallery,
    }


def write_events_csv(events: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_fields = sorted(
        {
            field
            for event in events
            for field in event
            if field.endswith("_capture_id")
            or field.endswith("_captured_at_utc")
            or field.endswith("_label")
            or field.endswith("_label_source")
            or field.endswith("_image_path")
        }
    )
    fieldnames = [
        "event_id",
        "event_time_utc",
        "event_time_local",
        "local_date",
        "local_hour",
        "local_decimal_hour",
        "event_label",
        "is_both_positive",
        "pair_delta_seconds",
        "label_pair",
        *camera_fields,
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(events)


def write_hour_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "local_hour",
                "event_count",
                "both_positive_count",
                "not_both_positive_count",
                "both_positive_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_hour_histogram(rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    hours = [int(row["local_hour"]) for row in rows]
    positives = [int(row["both_positive_count"]) for row in rows]
    totals = [int(row["event_count"]) for row in rows]
    negatives = [total - positive for total, positive in zip(totals, positives)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(hours, negatives, label="not both positive", color="#d0d7de")
    ax.bar(hours, positives, bottom=negatives, label=POSITIVE_EVENT_LABEL, color="#2da44e")
    ax.set_xticks(hours)
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Paired events")
    ax.set_title("Paired Mount Tam events by local hour")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_positive_event_gallery(
    events: list[dict[str, Any]],
    output_dir: Path,
    *,
    crop_pixels: PixelCrop | dict[str, int] | None,
    thumbnail_width: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_events = [row for row in events if row["event_label"] == POSITIVE_EVENT_LABEL]
    written = []
    skipped: Counter[str] = Counter()
    for index, event in enumerate(positive_events, start=1):
        try:
            path = write_side_by_side_image(
                event,
                output_dir / f"{index:04d}_{event['event_id']}.jpg",
                crop_pixels=crop_pixels,
                thumbnail_width=thumbnail_width,
            )
        except OSError:
            skipped["image_open_failed"] += 1
            continue
        written.append((event, path))
    write_gallery_html(written, output_dir / "index.html")
    return {
        "positive_event_count": len(positive_events),
        "side_by_side_images_written": len(written),
        "skipped": dict(sorted(skipped.items())),
        "index_html": str(output_dir / "index.html"),
    }


def write_side_by_side_image(
    event: dict[str, Any],
    output_path: Path,
    *,
    crop_pixels: PixelCrop | dict[str, int] | None,
    thumbnail_width: int,
) -> Path:
    image_fields = sorted(field for field in event if field.endswith("_image_path"))
    if len(image_fields) != 2:
        raise OSError("Expected exactly two image paths for paired event.")
    panels = []
    for image_field in image_fields:
        with Image.open(event[image_field]) as image:
            image = image.convert("RGB")
            if crop_to_dict(crop_pixels):
                image = crop_image(image, crop_pixels)
            panels.append(resize_to_width(image, thumbnail_width))

    caption_height = 42
    gutter = 8
    width = sum(panel.width for panel in panels) + gutter
    height = max(panel.height for panel in panels) + caption_height
    canvas = Image.new("RGB", (width, height), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + gutter
    caption = f"{event['event_time_local']}  |  {event['label_pair']}"
    draw.text((10, max(panel.height for panel in panels) + 12), caption, fill=(20, 20, 20))
    canvas.save(output_path, quality=92)
    return output_path


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    scale = width / image.width
    height = max(1, int(math.ceil(image.height * scale)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_gallery_html(items: list[tuple[dict[str, Any], Path]], output_path: Path) -> None:
    lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8">',
        "<title>Both-camera clouds-below-peak examples</title>",
        "<style>",
        "body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; background: #f6f8fa; }",
        ".event { margin: 0 0 28px; padding: 16px; background: white; border: 1px solid #d0d7de; border-radius: 10px; }",
        ".event img { max-width: 100%; border-radius: 8px; }",
        ".meta { color: #57606a; margin: 8px 0 0; font-size: 14px; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Both-camera clouds-below-peak examples</h1>",
        f"<p>{len(items)} paired positive event(s).</p>",
    ]
    for event, path in items:
        rel = path.name
        lines.extend(
            [
                '<div class="event">',
                f'<img src="{html.escape(rel)}" alt="{html.escape(event["event_id"])}">',
                (
                    '<div class="meta">'
                    f'{html.escape(event["event_time_local"])} · '
                    f'{html.escape(event["label_pair"])} · '
                    f'delta {html.escape(str(event["pair_delta_seconds"]))} sec'
                    "</div>"
                ),
                "</div>",
            ]
        )
    lines.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_summary_markdown(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Paired event summary: `{summary['task_id']}`",
        "",
        f"- Cameras: `{summary['camera_ids'][0]}` + `{summary['camera_ids'][1]}`",
        f"- Positive single-image label: `{summary['positive_label']}`",
        f"- Positive paired-event label: `{summary['event_positive_label']}`",
        f"- Max pairing window: {summary['max_pair_minutes']} minutes",
        f"- Timezone: `{summary['timezone']}`",
        f"- Paired events: {summary['event_count']}",
        f"- Both-camera positive events: {summary['both_positive_count']}",
        f"- Both-camera positive rate: {format_rate(summary['both_positive_rate'])}",
        "",
        "## Outputs",
        "",
        f"- Paired events CSV: `{summary['paths']['events_csv']}`",
        f"- Hour histogram CSV: `{summary['paths']['hour_csv']}`",
        f"- Hour histogram PNG: `{summary['paths']['hour_png']}`",
        f"- Positive example gallery: `{summary['paths']['examples_html']}`",
        "",
        "## Event labels",
        "",
    ]
    lines.extend(f"- `{label}`: {count}" for label, count in summary["event_label_counts"].items())
    lines.extend(["", "## Top local hours for both-camera positive events", ""])
    if summary["top_positive_hours"]:
        lines.extend(
            f"- {row['local_hour']:02d}:00 — {row['both_positive_count']}"
            for row in summary["top_positive_hours"][:10]
        )
    else:
        lines.append("- No positive paired events found.")
    lines.extend(["", "## Single-image label pairs", ""])
    lines.extend(f"- `{pair}`: {count}" for pair, count in summary["single_label_pair_counts"].items())
    lines.extend(["", "## Skipped", ""])
    lines.append(f"- Frames: `{summary['skipped_frames']}`")
    lines.append(f"- Pairs: `{summary['skipped_pairs']}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def safe_divide(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def format_rate(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1%}"


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
