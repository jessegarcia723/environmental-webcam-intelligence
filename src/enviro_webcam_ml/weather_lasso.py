from __future__ import annotations

import csv
import json
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from enviro_webcam_ml.image_training import binary_metrics


@dataclass(frozen=True)
class WeatherLassoOptions:
    training_csv: Path
    output_dir: Path
    positive_label: str
    features: tuple[str, ...] = ()
    max_weather_age_minutes: float = 90.0
    c: float = 1.0
    positive_threshold: float = 0.5
    class_weight: str | None = None
    split_strategy: str = "training_csv"
    blocked_val_fraction: float = 0.15
    blocked_test_fraction: float = 0.15
    random_state: int = 42


def train_weather_lasso(
    conn: sqlite3.Connection,
    options: WeatherLassoOptions,
) -> dict[str, Any]:
    rows = read_weather_training_rows(options.training_csv)
    if not rows:
        raise ValueError(f"No rows found in training CSV: {options.training_csv}")
    if options.positive_label not in {row["label"] for row in rows}:
        raise ValueError(f"positive_label={options.positive_label!r} is not present in the training CSV.")
    if not 0 <= options.positive_threshold <= 1:
        raise ValueError("positive_threshold must be between 0 and 1.")
    if options.c <= 0:
        raise ValueError("--c must be greater than 0.")
    validate_split_strategy(options.split_strategy)
    validate_blocked_fractions(options.blocked_val_fraction, options.blocked_test_fraction)

    weather = weather_records_by_camera(conn)
    examples, skipped = build_weather_examples(
        rows,
        weather,
        requested_features=options.features,
        max_weather_age_minutes=options.max_weather_age_minutes,
        positive_label=options.positive_label,
    )
    if not examples:
        raise ValueError(
            "No training rows could be matched to weather records. "
            "Run envirocam fetch-weather/run-collector, or increase --max-weather-age-minutes."
        )
    if options.split_strategy == "weather-hour-blocked":
        blocked_split_summary = assign_weather_hour_blocked_splits(
            examples,
            val_fraction=options.blocked_val_fraction,
            test_fraction=options.blocked_test_fraction,
        )
    else:
        blocked_split_summary = None

    feature_names = options.features or inferred_feature_names(examples)
    if not feature_names:
        raise ValueError("No numeric weather features were found in matched weather records.")

    train_examples = [example for example in examples if example["split"] == "train"]
    if not train_examples:
        raise ValueError("No train rows with matched weather records.")
    if len({example["target"] for example in train_examples}) < 2:
        raise ValueError("Train split needs both positive and non-positive weather examples.")

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Weather LASSO training requires scikit-learn. Install with: "
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
    pipeline.fit(
        feature_matrix(train_examples, feature_names),
        [example["target"] for example in train_examples],
    )

    prediction_rows = collect_weather_predictions(
        pipeline,
        examples,
        feature_names,
        positive_label=options.positive_label,
        positive_threshold=options.positive_threshold,
    )
    detailed_metrics = {
        split: weather_binary_report(
            [row for row in prediction_rows if row["split"] == split],
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
            },
            f,
        )

    summary = {
        "model_name": "weather_lasso_logistic",
        "training_csv": str(options.training_csv),
        "output_dir": str(options.output_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "coefficients_path": str(coefficients_path),
        "positive_label": options.positive_label,
        "positive_threshold": options.positive_threshold,
        "max_weather_age_minutes": options.max_weather_age_minutes,
        "c": options.c,
        "class_weight": class_weight or "none",
        "split_strategy": options.split_strategy,
        "blocked_val_fraction": options.blocked_val_fraction,
        "blocked_test_fraction": options.blocked_test_fraction,
        "features": feature_names,
        "nonzero_coefficients": [row for row in coefficients if row["coefficient"] != 0.0],
        "split_counts": split_counts(examples),
        "matched_rows": len(examples),
        "skipped": skipped,
        "blocked_split_summary": blocked_split_summary,
        "weather_group_leakage": weather_group_leakage(examples),
        "detailed_metrics": detailed_metrics,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_weather_predictions(prediction_rows, predictions_path)
    write_coefficients(coefficients, coefficients_path)
    return summary


def read_weather_training_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row.get("capture_id") and row.get("camera_id") and row.get("captured_at_utc") and row.get("label") and row.get("split")
    ]


def weather_records_by_camera(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    rows = conn.execute(
        """
        SELECT camera_id, valid_at_utc, fetched_at_utc, variables_json
        FROM weather_record
        ORDER BY camera_id, valid_at_utc, fetched_at_utc
        """
    ).fetchall()
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["camera_id"], row["valid_at_utc"])
        existing = latest_by_key.get(key)
        if existing is None or row["fetched_at_utc"] >= existing["fetched_at_utc"]:
            latest_by_key[key] = {
                "camera_id": row["camera_id"],
                "valid_at_utc": row["valid_at_utc"],
                "valid_at": parse_datetime(row["valid_at_utc"]),
                "fetched_at_utc": row["fetched_at_utc"],
                "variables": json.loads(row["variables_json"] or "{}"),
            }
    by_camera: dict[str, list[dict[str, Any]]] = {}
    for item in latest_by_key.values():
        by_camera.setdefault(item["camera_id"], []).append(item)
    for items in by_camera.values():
        items.sort(key=lambda item: item["valid_at"])
    return by_camera


def build_weather_examples(
    rows: list[dict[str, str]],
    weather: dict[str, list[dict[str, Any]]],
    *,
    requested_features: tuple[str, ...],
    max_weather_age_minutes: float,
    positive_label: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples = []
    skipped = {
        "no_weather_for_camera": 0,
        "no_weather_within_window": 0,
        "no_numeric_features": 0,
    }
    for row in rows:
        camera_weather = weather.get(row["camera_id"], [])
        if not camera_weather:
            skipped["no_weather_for_camera"] += 1
            continue
        captured_at = parse_datetime(row["captured_at_utc"])
        match = nearest_weather_record(camera_weather, captured_at)
        if match is None:
            skipped["no_weather_for_camera"] += 1
            continue
        age_minutes = abs((captured_at - match["valid_at"]).total_seconds()) / 60.0
        if age_minutes > max_weather_age_minutes:
            skipped["no_weather_within_window"] += 1
            continue
        features = numeric_weather_features(match["variables"], requested_features)
        if not features:
            skipped["no_numeric_features"] += 1
            continue
        examples.append(
            {
                "capture_id": row["capture_id"],
                "camera_id": row["camera_id"],
                "captured_at_utc": row["captured_at_utc"],
                "image_path": row.get("image_path", ""),
                "weather_valid_at_utc": match["valid_at_utc"],
                "weather_group": weather_group_key(row["camera_id"], match["valid_at_utc"]),
                "weather_age_minutes": age_minutes,
                "split": row["split"],
                "label": row["label"],
                "target": int(row["label"] == positive_label),
                "features": features,
            }
        )
    return examples, {key: value for key, value in skipped.items() if value}


def assign_weather_hour_blocked_splits(
    examples: list[dict[str, Any]],
    *,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    train_fraction = 1.0 - val_fraction - test_fraction
    groups = grouped_weather_examples(examples)
    group_rows = []
    for key, group_examples in groups.items():
        first = min(group_examples, key=lambda item: (item["captured_at_utc"], item["camera_id"], item["capture_id"]))
        group_rows.append(
            {
                "weather_group": key,
                "first_captured_at_utc": first["captured_at_utc"],
                "camera_id": first["camera_id"],
                "weather_valid_at_utc": first["weather_valid_at_utc"],
                "target": int(any(example["target"] for example in group_examples)),
                "row_count": len(group_examples),
            }
        )

    split_by_group: dict[str, str] = {}
    for target in (0, 1):
        target_groups = [group for group in group_rows if group["target"] == target]
        target_groups.sort(key=lambda item: (item["first_captured_at_utc"], item["camera_id"], item["weather_group"]))
        assigned = chronological_split_names(
            len(target_groups),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        for group, split in zip(target_groups, assigned):
            split_by_group[group["weather_group"]] = split

    for example in examples:
        example["original_split"] = example["split"]
        example["split"] = split_by_group[example["weather_group"]]

    return {
        "group_count": len(group_rows),
        "group_split_counts": counts_by_split(
            [{"split": split, "row_count": 1} for split in split_by_group.values()]
        ),
        "row_split_counts": split_counts(examples),
        "positive_group_count": sum(1 for group in group_rows if group["target"]),
        "non_positive_group_count": sum(1 for group in group_rows if not group["target"]),
    }


def grouped_weather_examples(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(example["weather_group"], []).append(example)
    return groups


def chronological_split_names(
    n: int,
    *,
    train_fraction: float,
    val_fraction: float,
) -> list[str]:
    if n <= 0:
        return []
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    if n >= 3:
        train_end = max(1, min(train_end, n - 2))
        val_end = max(train_end + 1, min(val_end, n - 1))
    elif n == 2:
        train_end = 1
        val_end = 1
    else:
        train_end = 1
        val_end = 1
    return [
        "train" if index < train_end else "val" if index < val_end else "test"
        for index in range(n)
    ]


def weather_group_leakage(examples: list[dict[str, Any]]) -> dict[str, Any]:
    groups = grouped_weather_examples(examples)
    split_sets = {key: {example["split"] for example in group} for key, group in groups.items()}
    train_groups = {key for key, splits in split_sets.items() if "train" in splits}
    leakage_rows = {
        split: sum(
            1
            for example in examples
            if example["split"] == split and example["weather_group"] in train_groups
        )
        for split in ("val", "test")
    }
    leaked_groups = {
        key: sorted(splits)
        for key, splits in split_sets.items()
        if len(splits) > 1
    }
    return {
        "unique_weather_groups": len(groups),
        "groups_spanning_multiple_splits": len(leaked_groups),
        "rows_sharing_weather_group_with_train": leakage_rows,
        "is_blocked": len(leaked_groups) == 0,
    }


def weather_group_key(camera_id: str, weather_valid_at_utc: str) -> str:
    return f"{camera_id}|{weather_valid_at_utc}"


def nearest_weather_record(records: list[dict[str, Any]], captured_at: datetime) -> dict[str, Any] | None:
    if not records:
        return None
    return min(records, key=lambda item: abs((captured_at - item["valid_at"]).total_seconds()))


def numeric_weather_features(variables: dict[str, Any], requested_features: tuple[str, ...]) -> dict[str, float]:
    names = requested_features or tuple(sorted(variables))
    features = {}
    for name in names:
        value = variables.get(name)
        if value in (None, ""):
            continue
        try:
            features[name] = float(value)
        except (TypeError, ValueError):
            continue
    return features


def inferred_feature_names(examples: list[dict[str, Any]]) -> list[str]:
    names = set()
    for example in examples:
        names.update(example["features"])
    return sorted(names)


def feature_matrix(examples: list[dict[str, Any]], feature_names: list[str]) -> list[list[float | None]]:
    return [
        [example["features"].get(name) for name in feature_names]
        for example in examples
    ]


def collect_weather_predictions(
    pipeline,
    examples: list[dict[str, Any]],
    feature_names: list[str],
    *,
    positive_label: str,
    positive_threshold: float,
) -> list[dict[str, Any]]:
    probabilities = pipeline.predict_proba(feature_matrix(examples, feature_names))[:, 1]
    rows = []
    negative_label = f"not_{positive_label}"
    for example, probability in zip(examples, probabilities):
        pred_label = positive_label if float(probability) >= positive_threshold else negative_label
        binary_true_label = positive_label if example["target"] else negative_label
        rows.append(
            {
                "split": example["split"],
                "capture_id": example["capture_id"],
                "camera_id": example["camera_id"],
                "captured_at_utc": example["captured_at_utc"],
                "weather_valid_at_utc": example["weather_valid_at_utc"],
                "weather_group": example["weather_group"],
                "weather_age_minutes": example["weather_age_minutes"],
                "true_label": example["label"],
                "binary_true_label": binary_true_label,
                "pred_label": pred_label,
                "positive_probability": float(probability),
                "correct": int(binary_true_label == pred_label),
            }
        )
    return rows


def weather_binary_report(predictions: list[dict[str, Any]], *, positive_label: str) -> dict[str, Any]:
    count = len(predictions)
    accuracy = None if count == 0 else sum(int(row["correct"]) for row in predictions) / count
    by_camera = {}
    for camera_id in sorted({row["camera_id"] for row in predictions}):
        camera_rows = [row for row in predictions if row["camera_id"] == camera_id]
        by_camera[camera_id] = {
            "accuracy": sum(int(row["correct"]) for row in camera_rows) / len(camera_rows),
            "count": len(camera_rows),
        }
    return {
        "overall": {"accuracy": accuracy, "count": count},
        "binary": binary_metrics(predictions, positive_label),
        "by_camera": by_camera,
    }


def coefficient_rows(pipeline, feature_names: list[str]) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    coefficients = model.coef_[0]
    rows = [
        {
            "feature": feature,
            "coefficient": float(coefficient),
            "abs_coefficient": abs(float(coefficient)),
        }
        for feature, coefficient in zip(feature_names, coefficients)
    ]
    return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)


def split_counts(examples: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for example in examples:
        split = example["split"]
        counts[split] = counts.get(split, 0) + 1
    return dict(sorted(counts.items()))


def counts_by_split(items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        split = item["split"]
        counts[split] = counts.get(split, 0) + int(item.get("row_count", 1))
    return dict(sorted(counts.items()))


def write_weather_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "split",
        "capture_id",
        "camera_id",
        "captured_at_utc",
        "weather_valid_at_utc",
        "weather_group",
        "weather_age_minutes",
        "true_label",
        "binary_true_label",
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


def validate_split_strategy(value: str) -> None:
    if value not in {"training_csv", "weather-hour-blocked"}:
        raise ValueError("--split-strategy must be either 'training_csv' or 'weather-hour-blocked'.")


def validate_blocked_fractions(val_fraction: float, test_fraction: float) -> None:
    if val_fraction < 0 or test_fraction < 0:
        raise ValueError("Blocked split fractions must be non-negative.")
    if val_fraction + test_fraction >= 1:
        raise ValueError("Blocked val/test fractions must leave some training groups.")


def parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
