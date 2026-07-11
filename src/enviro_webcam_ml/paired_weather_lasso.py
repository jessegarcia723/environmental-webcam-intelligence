from __future__ import annotations

import csv
import json
import pickle
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enviro_webcam_ml.image_training import binary_metrics
from enviro_webcam_ml.paired_events import NEGATIVE_EVENT_LABEL, POSITIVE_EVENT_LABEL
from enviro_webcam_ml.weather_lasso import (
    coefficient_rows,
    chronological_split_names,
    feature_matrix,
    inferred_feature_names,
    nearest_weather_record,
    numeric_weather_features,
    parse_datetime,
    validate_blocked_fractions,
    weather_records_by_camera,
)


@dataclass(frozen=True)
class PairedWeatherLassoOptions:
    paired_events_csv: Path
    output_dir: Path
    camera_ids: tuple[str, str]
    positive_label: str = POSITIVE_EVENT_LABEL
    features: tuple[str, ...] = ()
    max_weather_age_minutes: float = 90.0
    c: float = 1.0
    positive_threshold: float = 0.5
    class_weight: str | None = None
    split_strategy: str = "weather-hour-blocked"
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int = 42


def train_paired_weather_lasso(
    conn: sqlite3.Connection,
    options: PairedWeatherLassoOptions,
) -> dict[str, Any]:
    if len(options.camera_ids) != 2:
        raise ValueError("Paired weather LASSO requires exactly two camera IDs.")
    if not 0 <= options.positive_threshold <= 1:
        raise ValueError("positive_threshold must be between 0 and 1.")
    if options.c <= 0:
        raise ValueError("--c must be greater than 0.")
    validate_paired_split_strategy(options.split_strategy)
    validate_blocked_fractions(options.val_fraction, options.test_fraction)

    rows = read_paired_event_rows(options.paired_events_csv)
    if not rows:
        raise ValueError(f"No paired event rows found in {options.paired_events_csv}")
    weather = weather_records_by_camera(conn)
    examples, skipped = build_paired_weather_examples(
        rows,
        weather,
        camera_ids=options.camera_ids,
        requested_features=options.features,
        max_weather_age_minutes=options.max_weather_age_minutes,
        positive_label=options.positive_label,
    )
    if not examples:
        raise ValueError(
            "No paired event rows could be matched to weather records. "
            "Run envirocam fetch-weather/run-collector, or increase --max-weather-age-minutes."
        )

    if options.split_strategy == "weather-hour-blocked":
        split_summary = assign_paired_weather_blocked_splits(
            examples,
            train_fraction=options.train_fraction,
            val_fraction=options.val_fraction,
        )
    else:
        split_summary = assign_paired_chronological_splits(
            examples,
            train_fraction=options.train_fraction,
            val_fraction=options.val_fraction,
        )

    feature_names = inferred_feature_names(examples)
    if not feature_names:
        raise ValueError("No numeric paired weather features were found.")
    train_examples = [example for example in examples if example["split"] == "train"]
    if not train_examples:
        raise ValueError("No train paired weather examples.")
    if len({example["target"] for example in train_examples}) < 2:
        raise ValueError("Train split needs both positive and non-positive paired weather examples.")

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Paired weather LASSO training requires scikit-learn. Install with: "
            'python -m pip install -e ".[dev,train]"'
        ) from exc

    class_weight = None if options.class_weight in (None, "", "none") else options.class_weight
    if class_weight not in (None, "balanced"):
        raise ValueError("--class-weight must be either 'none' or 'balanced'.")

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    l1_ratio=1.0,
                    C=options.c,
                    class_weight=class_weight,
                    random_state=options.random_state,
                    max_iter=1000,
                ),
            ),
        ]
    )
    pipeline.fit(feature_matrix(train_examples, feature_names), [example["target"] for example in train_examples])
    predictions = collect_paired_weather_predictions(
        pipeline,
        examples,
        feature_names,
        positive_label=options.positive_label,
        positive_threshold=options.positive_threshold,
    )
    detailed_metrics = {
        split: paired_weather_report(
            [row for row in predictions if row["split"] == split],
            positive_label=options.positive_label,
        )
        for split in ("train", "val", "test")
    }

    options.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = options.output_dir / "model.pkl"
    metadata_path = options.output_dir / "metadata.json"
    predictions_path = options.output_dir / "predictions.csv"
    coefficients_path = options.output_dir / "coefficients.csv"
    coefficients = coefficient_rows(pipeline, feature_names)
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "pipeline": pipeline,
                "feature_names": feature_names,
                "positive_label": options.positive_label,
                "positive_threshold": options.positive_threshold,
                "camera_ids": options.camera_ids,
            },
            f,
        )
    summary = {
        "model_name": "weather_lasso_logistic",
        "model_type": "paired_weather_lasso",
        "event_scope": "paired_event",
        "paired_events_csv": str(options.paired_events_csv),
        "output_dir": str(options.output_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "coefficients_path": str(coefficients_path),
        "camera_ids": list(options.camera_ids),
        "positive_label": options.positive_label,
        "positive_threshold": options.positive_threshold,
        "max_weather_age_minutes": options.max_weather_age_minutes,
        "c": options.c,
        "class_weight": class_weight or "none",
        "split_strategy": options.split_strategy,
        "features": feature_names,
        "nonzero_coefficients": [row for row in coefficients if row["coefficient"] != 0.0],
        "split_counts": split_counts(examples),
        "matched_rows": len(examples),
        "skipped": skipped,
        "blocked_split_summary": split_summary,
        "weather_group_leakage": paired_weather_group_leakage(examples),
        "detailed_metrics": detailed_metrics,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_paired_weather_predictions(predictions, predictions_path)
    write_coefficients(coefficients, coefficients_path)
    return summary


def read_paired_event_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row for row in rows
        if row.get("event_id") and row.get("event_time_utc") and row.get("event_label")
    ]


def build_paired_weather_examples(
    rows: list[dict[str, str]],
    weather: dict[str, list[dict[str, Any]]],
    *,
    camera_ids: tuple[str, str],
    requested_features: tuple[str, ...],
    max_weather_age_minutes: float,
    positive_label: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples = []
    skipped: Counter[str] = Counter()
    for row in rows:
        features: dict[str, float] = {}
        weather_groups = []
        weather_valid = []
        ages = []
        for camera_id in camera_ids:
            captured_value = row.get(f"{camera_id}_captured_at_utc") or row.get("event_time_utc") or ""
            camera_weather = weather.get(camera_id, [])
            if not camera_weather:
                skipped[f"no_weather_for_{camera_id}"] += 1
                break
            captured_at = parse_datetime(captured_value)
            match = nearest_weather_record(camera_weather, captured_at)
            if match is None:
                skipped[f"no_weather_for_{camera_id}"] += 1
                break
            age_minutes = abs((captured_at - match["valid_at"]).total_seconds()) / 60.0
            if age_minutes > max_weather_age_minutes:
                skipped[f"no_weather_within_window_for_{camera_id}"] += 1
                break
            camera_features = numeric_weather_features(match["variables"], requested_features)
            if not camera_features:
                skipped[f"no_numeric_features_for_{camera_id}"] += 1
                break
            prefix = safe_feature_prefix(camera_id)
            for name, value in camera_features.items():
                features[f"{prefix}__{name}"] = value
            weather_groups.append(f"{camera_id}|{match['valid_at_utc']}")
            weather_valid.append(f"{camera_id}:{match['valid_at_utc']}")
            ages.append(age_minutes)
        else:
            if not features:
                skipped["no_features"] += 1
                continue
            label = row["event_label"]
            examples.append(
                {
                    "event_id": row["event_id"],
                    "capture_id": row["event_id"],
                    "camera_id": "paired",
                    "captured_at_utc": row["event_time_utc"],
                    "event_time_local": row.get("event_time_local", ""),
                    "event_time_local_label": row.get("event_time_local_label", ""),
                    "weather_valid_at_utc": "|".join(weather_valid),
                    "weather_group": "||".join(weather_groups),
                    "weather_age_minutes": max(ages),
                    "split": row.get("split") or "",
                    "label": label,
                    "target": int(label == positive_label),
                    "features": features,
                }
            )
    return examples, dict(sorted(skipped.items()))


def assign_paired_chronological_splits(
    examples: list[dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    split_by_id = {}
    for target in (0, 1):
        target_examples = [example for example in examples if example["target"] == target]
        target_examples.sort(key=lambda row: (row["captured_at_utc"], row["event_id"]))
        splits = chronological_split_names(len(target_examples), train_fraction=train_fraction, val_fraction=val_fraction)
        for example, split in zip(target_examples, splits):
            split_by_id[example["event_id"]] = split
    for example in examples:
        example["original_split"] = example.get("split", "")
        example["split"] = split_by_id[example["event_id"]]
    return {
        "row_split_counts": split_counts(examples),
        "positive_count": sum(1 for example in examples if example["target"]),
        "non_positive_count": sum(1 for example in examples if not example["target"]),
    }


def assign_paired_weather_blocked_splits(
    examples: list[dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(example["weather_group"], []).append(example)
    group_rows = []
    for key, group_examples in groups.items():
        first = min(group_examples, key=lambda row: (row["captured_at_utc"], row["event_id"]))
        group_rows.append(
            {
                "weather_group": key,
                "first_captured_at_utc": first["captured_at_utc"],
                "target": int(any(example["target"] for example in group_examples)),
                "row_count": len(group_examples),
            }
        )
    split_by_group = {}
    for target in (0, 1):
        target_groups = [row for row in group_rows if row["target"] == target]
        target_groups.sort(key=lambda row: (row["first_captured_at_utc"], row["weather_group"]))
        splits = chronological_split_names(len(target_groups), train_fraction=train_fraction, val_fraction=val_fraction)
        for group, split in zip(target_groups, splits):
            split_by_group[group["weather_group"]] = split
    for example in examples:
        example["original_split"] = example.get("split", "")
        example["split"] = split_by_group[example["weather_group"]]
    return {
        "group_count": len(group_rows),
        "group_split_counts": Counter(split_by_group.values()),
        "row_split_counts": split_counts(examples),
        "positive_group_count": sum(1 for row in group_rows if row["target"]),
        "non_positive_group_count": sum(1 for row in group_rows if not row["target"]),
    }


def collect_paired_weather_predictions(
    pipeline,
    examples: list[dict[str, Any]],
    feature_names: list[str],
    *,
    positive_label: str,
    positive_threshold: float,
) -> list[dict[str, Any]]:
    probabilities = pipeline.predict_proba(feature_matrix(examples, feature_names))[:, 1]
    rows = []
    for example, probability in zip(examples, probabilities):
        pred_label = positive_label if float(probability) >= positive_threshold else NEGATIVE_EVENT_LABEL
        true_label = positive_label if example["target"] else NEGATIVE_EVENT_LABEL
        rows.append(
            {
                "split": example["split"],
                "event_id": example["event_id"],
                "event_time_utc": example["captured_at_utc"],
                "event_time_local": example.get("event_time_local", ""),
                "event_time_local_label": example.get("event_time_local_label", ""),
                "weather_valid_at_utc": example["weather_valid_at_utc"],
                "weather_group": example["weather_group"],
                "weather_age_minutes": example["weather_age_minutes"],
                "true_label": true_label,
                "pred_label": pred_label,
                "positive_probability": float(probability),
                "correct": int(true_label == pred_label),
            }
        )
    return rows


def paired_weather_report(predictions: list[dict[str, Any]], *, positive_label: str) -> dict[str, Any]:
    count = len(predictions)
    accuracy = None if count == 0 else sum(int(row["correct"]) for row in predictions) / count
    return {
        "overall": {"accuracy": accuracy, "count": count},
        "binary": binary_metrics(predictions, positive_label),
        "by_camera": {},
    }


def paired_weather_group_leakage(examples: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    for example in examples:
        groups.setdefault(example["weather_group"], set()).add(example["split"])
    leaked = {key: splits for key, splits in groups.items() if len(splits) > 1}
    train_groups = {key for key, splits in groups.items() if "train" in splits}
    return {
        "unique_weather_groups": len(groups),
        "groups_spanning_multiple_splits": len(leaked),
        "rows_sharing_weather_group_with_train": {
            split: sum(
                1
                for example in examples
                if example["split"] == split and example["weather_group"] in train_groups
            )
            for split in ("val", "test")
        },
        "is_blocked": len(leaked) == 0,
    }


def split_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(example["split"] for example in examples)
    return dict(sorted(counts.items()))


def write_paired_weather_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "split",
        "event_id",
        "event_time_utc",
        "event_time_local",
        "event_time_local_label",
        "weather_valid_at_utc",
        "weather_group",
        "weather_age_minutes",
        "true_label",
        "pred_label",
        "positive_probability",
        "correct",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)


def write_coefficients(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "coefficient", "abs_coefficient"])
        writer.writeheader()
        writer.writerows(rows)


def validate_paired_split_strategy(value: str) -> None:
    if value not in {"chronological", "weather-hour-blocked"}:
        raise ValueError("--split-strategy must be either 'chronological' or 'weather-hour-blocked'.")


def safe_feature_prefix(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")
