from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from enviro_webcam_ml.config import AppConfig
from enviro_webcam_ml.model_comparison import format_metric


def build_study_report(
    *,
    config: AppConfig,
    task_id: str,
    output_dir: Path,
    training_csv: Path | None = None,
    paired_events_csv: Path | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    task = config.task(task_id)
    positive_label = config.task_positive_label(task_id) or str(task.get("positive_label") or "positive")
    training_csv = training_csv or config.task_training_csv_path(task_id)
    models_dir = models_dir or config.task_model_dir(task_id)
    paired_events_csv = paired_events_csv or config.data_dir / "reports" / "paired_events" / "paired_events.csv"
    timezone_name = config.cameras[0].location.timezone if config.cameras else "UTC"

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "study_report.md"
    model_comparison_csv = output_dir / "study_model_comparison.csv"
    single_hour_csv = output_dir / "single_camera_positive_hour_histogram.csv"
    hour_plot_png = output_dir / "event_hour_comparison.png"

    single_rows = read_csv_rows(training_csv)
    paired_rows = read_csv_rows(paired_events_csv)
    model_rows = model_summary_rows(models_dir)
    single_hour_rows = single_camera_hour_rows(
        single_rows,
        positive_label=positive_label,
        timezone_name=timezone_name,
    )
    paired_hour_rows = paired_event_hour_rows(paired_rows)
    write_csv(single_hour_rows, single_hour_csv)
    write_csv(model_rows, model_comparison_csv)
    plot_event_hours(single_hour_rows, paired_hour_rows, hour_plot_png)

    summary = {
        "task_id": task_id,
        "positive_label": positive_label,
        "training_csv": str(training_csv),
        "paired_events_csv": str(paired_events_csv),
        "models_dir": str(models_dir),
        "report_path": str(report_path),
        "model_comparison_csv": str(model_comparison_csv),
        "single_hour_csv": str(single_hour_csv),
        "hour_plot_png": str(hour_plot_png),
        "single_rows": len(single_rows),
        "paired_rows": len(paired_rows),
        "model_rows": len(model_rows),
    }
    write_report_markdown(
        report_path,
        summary=summary,
        single_hour_rows=single_hour_rows,
        paired_hour_rows=paired_hour_rows,
        model_rows=model_rows,
        weather_sections=weather_lasso_sections(model_rows),
        single_model_sections=single_model_sections(model_rows),
        camera_comparison=camera_comparison_section(model_rows),
    )
    return summary


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def model_summary_rows(models_dir: Path) -> list[dict[str, Any]]:
    if not models_dir.exists():
        return []
    rows = []
    for metadata_path in sorted(models_dir.rglob("metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if metadata_path.parent.name == "weather_lasso_feature_selection":
            experiment_role = "feature_selection"
        else:
            experiment_role = "model"
        row = model_summary_row(metadata_path, metadata, experiment_role=experiment_role)
        run = str(metadata_path.parent.relative_to(models_dir))
        row["run"] = run
        row["relative_run"] = run
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["experiment_role"] == "model",
            row["blocked"],
            row["test_accuracy"] if isinstance(row["test_accuracy"], float) else -1,
        ),
        reverse=True,
    )
    return rows


def model_summary_row(metadata_path: Path, metadata: dict[str, Any], *, experiment_role: str) -> dict[str, Any]:
    detailed = metadata.get("detailed_metrics", {})
    test = detailed.get("test", {})
    test_overall = test.get("overall", {})
    test_binary = test.get("binary", {})
    model_name = metadata.get("model_name") or "unknown"
    model_type = metadata.get("model_type") or ""
    run = str(metadata_path.parent)
    category = model_category(metadata_path, metadata)
    row = {
        "run": metadata_path.parent.name,
        "relative_run": str(metadata_path.parent),
        "category": category,
        "experiment_role": experiment_role,
        "model_name": model_name,
        "model_type": model_type,
        "split_strategy": metadata.get("split_strategy") or "training_csv",
        "blocked": ("blocked" in str(metadata.get("split_strategy") or "")),
        "test_accuracy": test_overall.get("accuracy"),
        "test_ppv": test_binary.get("ppv"),
        "test_sensitivity": test_binary.get("sensitivity"),
        "test_specificity": test_binary.get("specificity"),
        "test_true_positive": test_binary.get("true_positive"),
        "test_false_positive": test_binary.get("false_positive"),
        "test_true_negative": test_binary.get("true_negative"),
        "test_false_negative": test_binary.get("false_negative"),
        "test_ppv_fraction": metric_fraction(
            test_binary.get("true_positive"),
            none_sum(test_binary.get("true_positive"), test_binary.get("false_positive")),
        ),
        "test_sensitivity_fraction": metric_fraction(
            test_binary.get("true_positive"),
            none_sum(test_binary.get("true_positive"), test_binary.get("false_negative")),
        ),
        "test_specificity_fraction": metric_fraction(
            test_binary.get("true_negative"),
            none_sum(test_binary.get("true_negative"), test_binary.get("false_positive")),
        ),
        "test_count": test_overall.get("count"),
        "positive_label": metadata.get("positive_label") or test_binary.get("positive_label"),
        "weather_features": ", ".join(metadata.get("weather_features") or metadata.get("features") or []),
        "weather_feature_count": len(metadata.get("weather_features") or metadata.get("features") or []),
        "forecast_horizon_hours": metadata.get("forecast_horizon_hours"),
        "forecast_horizon_tolerance_minutes": metadata.get("horizon_tolerance_minutes"),
        "nonzero_coefficients": coefficient_summary(metadata),
        "metadata_path": str(metadata_path),
        "predictions_path": str(metadata.get("predictions_path") or ""),
    }
    for camera_id, metrics in test.get("by_camera", {}).items():
        row[f"test_camera_{camera_id}_accuracy"] = metrics.get("accuracy")
        row[f"test_camera_{camera_id}_count"] = metrics.get("count")
    if metadata.get("event_scope") == "paired_event" or "paired" in run or row["positive_label"] == "both_cameras_clouds_below_peak":
        row["event_scope"] = "paired_event"
    else:
        row["event_scope"] = "single_image"
    return row


def model_category(metadata_path: Path, metadata: dict[str, Any]) -> str:
    model_name = metadata.get("model_name")
    model_type = metadata.get("model_type")
    path_text = str(metadata_path)
    if metadata_path.parent.name == "weather_lasso_feature_selection":
        return "weather_lasso_feature_selection"
    if model_type == "forecast_weather_lasso" or model_name == "forecast_weather_lasso_logistic":
        return "forecast_weather_lasso"
    if model_name == "weather_lasso_logistic":
        return "weather_lasso"
    if model_type == "weather_classifier":
        return "weather_only"
    if model_type == "image_weather_fusion":
        if "lasso_selected" in path_text:
            return "image_plus_lasso_weather"
        return "image_plus_weather"
    return "image_only"


def coefficient_summary(metadata: dict[str, Any]) -> str:
    rows = metadata.get("nonzero_coefficients") or []
    if not rows:
        rows = metadata.get("feature_importances") or []
    if not rows:
        return ""
    return "; ".join(
        f"{row.get('feature')}={float(row.get('coefficient', 0.0)):.3g}"
        for row in rows
    )


def none_sum(*values: Any) -> int | None:
    if any(value is None for value in values):
        return None
    return sum(int(value) for value in values)


def metric_fraction(numerator: Any, denominator: Any) -> str:
    if numerator is None or denominator is None:
        return ""
    return f"{int(numerator)}/{int(denominator)}"


def count_text(value: Any) -> str:
    if value is None:
        return ""
    return str(int(value))


def feature_count_text(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    return str(int(value))


def horizon_text(value: Any) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):g}h"


def single_camera_hour_rows(
    rows: list[dict[str, str]],
    *,
    positive_label: str,
    timezone_name: str,
) -> list[dict[str, Any]]:
    tz = ZoneInfo(timezone_name)
    cameras = sorted({row.get("camera_id", "") for row in rows if row.get("camera_id")})
    counts: dict[tuple[str, int], Counter[str]] = {
        (camera, hour): Counter() for camera in cameras for hour in range(24)
    }
    for row in rows:
        camera = row.get("camera_id", "")
        if not camera:
            continue
        captured_at = parse_datetime(row.get("captured_at_utc", ""))
        if captured_at is None:
            continue
        hour = captured_at.astimezone(tz).hour
        counts[(camera, hour)]["total"] += 1
        if row.get("label") == positive_label:
            counts[(camera, hour)]["positive"] += 1
    result = []
    for camera in cameras:
        for hour in range(24):
            counter = counts[(camera, hour)]
            total = counter["total"]
            positive = counter["positive"]
            result.append(
                {
                    "camera_id": camera,
                    "local_hour": hour,
                    "single_frame_count": total,
                    "positive_count": positive,
                    "positive_rate": positive / total if total else "",
                }
            )
    return result


def paired_event_hour_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    counts = {hour: Counter() for hour in range(24)}
    for row in rows:
        try:
            hour = int(row.get("local_hour", ""))
        except ValueError:
            continue
        counts[hour]["total"] += 1
        if row.get("is_both_positive") in ("1", "True", "true"):
            counts[hour]["positive"] += 1
    result = []
    for hour in range(24):
        total = counts[hour]["total"]
        positive = counts[hour]["positive"]
        result.append(
            {
                "local_hour": hour,
                "paired_event_count": total,
                "both_positive_count": positive,
                "both_positive_rate": positive / total if total else "",
            }
        )
    return result


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_event_hours(
    single_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    hours = list(range(24))
    for camera in sorted({row["camera_id"] for row in single_rows}):
        positives = [
            int(next((row["positive_count"] for row in single_rows if row["camera_id"] == camera and row["local_hour"] == hour), 0))
            for hour in hours
        ]
        ax.plot(hours, positives, marker="o", label=f"{camera} single positive")
    paired = [
        int(next((row["both_positive_count"] for row in paired_rows if row["local_hour"] == hour), 0))
        for hour in hours
    ]
    ax.bar(hours, paired, alpha=0.35, label="both cameras positive")
    ax.set_xticks(hours)
    ax.set_xlabel("Local hour")
    ax.set_ylabel("Positive examples")
    ax.set_title("Single-camera and paired-event positives by local hour")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def weather_lasso_sections(model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    weather_categories = {"weather_lasso", "weather_only"}
    forecast_categories = {"forecast_weather_lasso"}
    single_weather = [
        row for row in model_rows
        if row["event_scope"] == "single_image"
        and row["category"] in weather_categories
        and row["experiment_role"] == "model"
    ]
    paired_weather = [
        row for row in model_rows
        if row["event_scope"] == "paired_event"
        and row["category"] in weather_categories
        and row["experiment_role"] == "model"
    ]
    forecast_weather = [
        row for row in model_rows
        if row["event_scope"] == "single_image"
        and row["category"] in forecast_categories
        and row["experiment_role"] == "model"
    ]
    return {
        "single_lasso": best_rows(single_weather),
        "paired_lasso": best_rows(paired_weather),
        "forecast_lasso": best_rows(forecast_weather),
    }


def single_model_sections(model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    single = [
        row for row in model_rows
        if row["event_scope"] == "single_image"
        and row["experiment_role"] == "model"
        and row["category"] in {"image_only", "image_plus_weather", "image_plus_lasso_weather"}
    ]
    paired = [
        row for row in model_rows
        if row["event_scope"] == "paired_event"
        and row["experiment_role"] == "model"
        and row["category"] in {"image_only", "image_plus_weather", "image_plus_lasso_weather"}
    ]
    return {
        "single_neural_models": best_rows(single),
        "paired_neural_models": best_rows(paired),
        "single_by_category": best_by_category(single),
        "paired_by_category": best_by_category(paired),
    }


def camera_comparison_section(model_rows: list[dict[str, Any]]) -> dict[str, Any]:
    shared_rows = [
        row for row in model_rows
        if row["event_scope"] == "single_image"
        and row["experiment_role"] == "model"
        and row["category"] in {"image_only", "image_plus_weather", "image_plus_lasso_weather"}
        and any(key.startswith("test_camera_") for key in row)
        and "camera_specific" not in row["relative_run"]
    ]
    separate_rows = [
        row for row in model_rows
        if row["event_scope"] == "single_image"
        and row["experiment_role"] == "model"
        and ("separate" in row["relative_run"] or "camera_specific" in row["relative_run"])
    ]
    return {
        "shared_models": best_rows(shared_rows),
        "separate_models": best_rows(separate_rows),
    }


def best_rows(rows: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    rows = [row for row in rows if row.get("test_accuracy") is not None]
    rows.sort(
        key=lambda row: (
            bool(row.get("blocked")),
            float(row.get("test_accuracy") or -1),
            float(row.get("test_specificity") or -1),
            float(row.get("test_ppv") or -1),
        ),
        reverse=True,
    )
    return rows[:limit]


def best_by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for category in sorted({row["category"] for row in rows}):
        ranked = best_rows([row for row in rows if row["category"] == category], limit=1)
        result.extend(ranked)
    result.sort(key=lambda row: float(row.get("test_accuracy") or -1), reverse=True)
    return result


def write_report_markdown(
    output_path: Path,
    *,
    summary: dict[str, Any],
    single_hour_rows: list[dict[str, Any]],
    paired_hour_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    weather_sections: dict[str, Any],
    single_model_sections: dict[str, Any],
    camera_comparison: dict[str, Any],
) -> None:
    lines = [
        f"# Environmental webcam study report: `{summary['task_id']}`",
        "",
        "This report summarizes already-generated datasets and model runs. It does not retrain models.",
        "",
        "## Files summarized",
        "",
        f"- Single-image training CSV: `{summary['training_csv']}`",
        f"- Paired-events CSV: `{summary['paired_events_csv']}`",
        f"- Models directory: `{summary['models_dir']}`",
        f"- Model comparison CSV: `{summary['model_comparison_csv']}`",
        f"- Single-camera hour histogram CSV: `{summary['single_hour_csv']}`",
        f"- Hour comparison plot: `{summary['hour_plot_png']}`",
        "",
        "## 1. Best times for single-camera and paired events",
        "",
        "### Paired both-camera events",
        "",
        *top_paired_hour_lines(paired_hour_rows),
        "",
        "### Single-camera positive events",
        "",
        *top_single_hour_lines(single_hour_rows),
        "",
        "## 2. Weather-only predictors and performance",
        "",
        "### Single-image event",
        "",
        *model_table_or_missing(weather_sections["single_lasso"], include_coefficients=True),
        "",
        "### Paired event",
        "",
        *model_table_or_missing(
            weather_sections["paired_lasso"],
            include_coefficients=True,
            missing_text=(
                "No paired-event weather-only model run was found yet. "
                "The paired-event dataset exists, but paired weather training has not been run yet."
            ),
        ),
        "",
        "### Forecast backtest",
        "",
        "These models use forecast rows known before the image time. The `Features` column matters: archived historical runs may have fewer usable non-null features than live-saved forecasts.",
        "",
        *model_table_or_missing(
            weather_sections["forecast_lasso"],
            include_coefficients=True,
            missing_text=(
                "No forecast-weather model runs were found yet. "
                "Run `envirocam backfill-historical-forecasts` for old annotations, "
                "or collect live forecasts going forward, then run the study suite."
            ),
        ),
        "",
        "## 3. Best neural-net classifiers",
        "",
        "### Single-image event",
        "",
        *model_table_or_missing(single_model_sections["single_neural_models"]),
        "",
        "### Paired event",
        "",
        *model_table_or_missing(
            single_model_sections["paired_neural_models"],
            missing_text=(
                "No paired-image neural-network runs were found yet. "
                "This would require a paired-image trainer that consumes east+west images together."
            ),
        ),
        "",
        "## 4. Image vs image+weather vs image+LASSO-selected weather",
        "",
        "### Single-image event: best run per category",
        "",
        *model_table_or_missing(single_model_sections["single_by_category"]),
        "",
        "### Paired event: best run per category",
        "",
        *model_table_or_missing(
            single_model_sections["paired_by_category"],
            missing_text="No paired image/image+weather model runs were found yet.",
        ),
        "",
        "## 5. Shared model vs separate camera-specific models",
        "",
        "### Shared models with by-camera test metrics",
        "",
        *shared_camera_lines(camera_comparison["shared_models"]),
        "",
        "### Separate camera-specific models",
        "",
        *model_table_or_missing(
            camera_comparison["separate_models"],
            missing_text=(
                "No separate camera-specific model runs were found. "
                "Current image models are shared models trained on east+west together, with camera_id used for reporting."
            ),
        ),
        "",
        "## All discovered model runs",
        "",
        *model_table_or_missing(model_rows[:25], include_coefficients=False),
        "",
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def top_paired_hour_lines(rows: list[dict[str, Any]]) -> list[str]:
    positives = [
        row for row in rows
        if int(row.get("both_positive_count") or 0) > 0
    ]
    positives.sort(key=lambda row: int(row.get("both_positive_count") or 0), reverse=True)
    if not positives:
        return ["No paired positive events were found."]
    return [
        "| Local hour | Both-positive count | Total paired events | Positive rate |",
        "| ---: | ---: | ---: | ---: |",
        *[
            "| "
            + " | ".join(
                [
                    f"{int(row['local_hour']):02d}:00",
                    str(row["both_positive_count"]),
                    str(row["paired_event_count"]),
                    format_rate(row["both_positive_rate"]),
                ]
            )
            + " |"
            for row in positives[:10]
        ],
    ]


def top_single_hour_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No single-camera training rows were found."]
    lines = [
        "| Camera | Local hour | Positive count | Total frames | Positive rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for camera in sorted({row["camera_id"] for row in rows}):
        camera_rows = [row for row in rows if row["camera_id"] == camera]
        camera_rows.sort(key=lambda row: int(row.get("positive_count") or 0), reverse=True)
        for row in camera_rows[:5]:
            if int(row.get("positive_count") or 0) == 0:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        f"`{camera}`",
                        f"{int(row['local_hour']):02d}:00",
                        str(row["positive_count"]),
                        str(row["single_frame_count"]),
                        format_rate(row["positive_rate"]),
                    ]
                )
                + " |"
            )
    if len(lines) == 2:
        lines.append("| No positive hours found |  |  |  |  |")
    return lines


def model_table_or_missing(
    rows: list[dict[str, Any]],
    *,
    include_coefficients: bool = False,
    missing_text: str = "No matching model runs were found.",
) -> list[str]:
    if not rows:
        return [missing_text]
    headers = [
        "Run",
        "Category",
        "Model",
        "Horizon",
        "Features",
        "Blocked",
        "Accuracy",
        "PPV",
        "PPV frac",
        "Sensitivity",
        "Sensitivity frac",
        "Specificity",
        "Specificity frac",
        "TP",
        "FP",
        "TN",
        "FN",
        "N",
    ]
    if include_coefficients:
        headers.append("Feature weights/importances")
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(
            [
                "---",
                "---",
                "---",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
                "---:",
            ]
            + (["---"] if include_coefficients else [])
        )
        + " |",
    ]
    for row in rows:
        values = [
            f"`{row['run']}`",
            f"`{row['category']}`",
            f"`{row['model_name']}`",
            horizon_text(row.get("forecast_horizon_hours")),
            feature_count_text(row.get("weather_feature_count")),
            "yes" if row.get("blocked") else "no",
            format_metric(row.get("test_accuracy")),
            format_metric(row.get("test_ppv")),
            row.get("test_ppv_fraction") or "",
            format_metric(row.get("test_sensitivity")),
            row.get("test_sensitivity_fraction") or "",
            format_metric(row.get("test_specificity")),
            row.get("test_specificity_fraction") or "",
            count_text(row.get("test_true_positive")),
            count_text(row.get("test_false_positive")),
            count_text(row.get("test_true_negative")),
            count_text(row.get("test_false_negative")),
            str(row.get("test_count") or ""),
        ]
        if include_coefficients:
            values.append(row.get("nonzero_coefficients") or "")
        lines.append("| " + " | ".join(values) + " |")
    return lines


def shared_camera_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No shared model by-camera metrics were found."]
    lines = model_table_or_missing(rows[:5])
    camera_keys = sorted(
        {
            key
            for row in rows[:5]
            for key in row
            if key.startswith("test_camera_") and key.endswith("_accuracy")
        }
    )
    if not camera_keys:
        return lines
    lines.extend(["", "| Run | Camera | Test accuracy | N |", "| --- | --- | ---: | ---: |"])
    for row in rows[:5]:
        for key in camera_keys:
            camera = key[len("test_camera_") : -len("_accuracy")]
            count = row.get(f"test_camera_{camera}_count") or ""
            lines.append(f"| `{row['run']}` | `{camera}` | {format_metric(row.get(key))} | {count} |")
    return lines


def format_rate(value: Any) -> str:
    if value in ("", None):
        return "n/a"
    return f"{float(value):.1%}"


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed
