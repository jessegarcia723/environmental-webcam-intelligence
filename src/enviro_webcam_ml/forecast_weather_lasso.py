from __future__ import annotations

import csv
import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enviro_webcam_ml.image_training import binary_metrics
from enviro_webcam_ml.weather_lasso import (
    chronological_split_names,
    coefficient_rows,
    feature_matrix,
    inferred_feature_names,
    numeric_weather_features,
    parse_datetime,
    read_weather_training_rows,
    split_counts,
    validate_blocked_fractions,
)


@dataclass(frozen=True)
class ForecastWeatherLassoOptions:
    training_csv: Path
    output_dir: Path
    positive_label: str
    features: tuple[str, ...] = ()
    forecast_horizon_hours: float = 3.0
    horizon_tolerance_minutes: float = 75.0
    max_weather_age_minutes: float = 90.0
    require_all_features: bool = True
    c: float = 1.0
    positive_threshold: float = 0.5
    class_weight: str | None = None
    split_strategy: str = "forecast-issue-blocked"
    blocked_val_fraction: float = 0.15
    blocked_test_fraction: float = 0.15
    random_state: int = 42


def train_forecast_weather_lasso(
    conn: sqlite3.Connection,
    options: ForecastWeatherLassoOptions,
) -> dict[str, Any]:
    validate_forecast_options(options)
    rows = read_weather_training_rows(options.training_csv)
    if not rows:
        raise ValueError(f"No rows found in training CSV: {options.training_csv}")
    if options.positive_label not in {row["label"] for row in rows}:
        raise ValueError(f"positive_label={options.positive_label!r} is not present in the training CSV.")

    weather, weather_sanity = forecast_weather_records_by_camera(conn)
    examples, skipped = build_forecast_weather_examples(
        rows,
        weather,
        requested_features=options.features,
        forecast_horizon_hours=options.forecast_horizon_hours,
        horizon_tolerance_minutes=options.horizon_tolerance_minutes,
        max_weather_age_minutes=options.max_weather_age_minutes,
        require_all_features=options.require_all_features,
        positive_label=options.positive_label,
    )
    if not examples:
        raise ValueError(
            "No training rows could be matched to forecast weather records. "
            "Run envirocam fetch-weather/run-collector for long enough to save future forecast rows, "
            "or increase --horizon-tolerance-minutes / --max-weather-age-minutes."
        )
    if options.split_strategy == "forecast-issue-blocked":
        split_summary = assign_forecast_issue_blocked_splits(
            examples,
            val_fraction=options.blocked_val_fraction,
            test_fraction=options.blocked_test_fraction,
        )
    else:
        split_summary = None

    feature_names = list(options.features or inferred_feature_names(examples))
    if not feature_names:
        raise ValueError("No numeric forecast weather features were found in matched weather records.")
    train_examples = [example for example in examples if example["split"] == "train"]
    if not train_examples:
        raise ValueError("No train rows with matched forecast weather records.")
    if len({example["target"] for example in train_examples}) < 2:
        raise ValueError("Train split needs both positive and non-positive forecast examples.")

    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Forecast weather LASSO training requires scikit-learn. Install with: "
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

    prediction_rows = collect_forecast_weather_predictions(
        pipeline,
        examples,
        feature_names,
        positive_label=options.positive_label,
        positive_threshold=options.positive_threshold,
    )
    detailed_metrics = {
        split: forecast_weather_report(
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
                "forecast_horizon_hours": options.forecast_horizon_hours,
            },
            f,
        )
    summary = {
        "model_name": "forecast_weather_lasso_logistic",
        "model_type": "forecast_weather_lasso",
        "event_scope": "single_image",
        "training_csv": str(options.training_csv),
        "output_dir": str(options.output_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "coefficients_path": str(coefficients_path),
        "positive_label": options.positive_label,
        "positive_threshold": options.positive_threshold,
        "forecast_horizon_hours": options.forecast_horizon_hours,
        "horizon_tolerance_minutes": options.horizon_tolerance_minutes,
        "max_weather_age_minutes": options.max_weather_age_minutes,
        "require_all_features": options.require_all_features,
        "c": options.c,
        "class_weight": class_weight or "none",
        "split_strategy": options.split_strategy,
        "features": feature_names,
        "nonzero_coefficients": [row for row in coefficients if row["coefficient"] != 0.0],
        "split_counts": split_counts(examples),
        "matched_rows": len(examples),
        "skipped": skipped,
        "weather_sanity": weather_sanity,
        "matched_sanity": matched_sanity(examples, feature_names),
        "blocked_split_summary": split_summary,
        "forecast_group_leakage": forecast_group_leakage(examples),
        "detailed_metrics": detailed_metrics,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_forecast_weather_predictions(prediction_rows, predictions_path)
    write_forecast_coefficients(coefficients, coefficients_path)
    return summary


def validate_forecast_options(options: ForecastWeatherLassoOptions) -> None:
    if options.forecast_horizon_hours <= 0:
        raise ValueError("--forecast-horizon-hours must be greater than 0 for real forecast backtests.")
    if options.horizon_tolerance_minutes < 0:
        raise ValueError("--horizon-tolerance-minutes must be non-negative.")
    if options.max_weather_age_minutes < 0:
        raise ValueError("--max-weather-age-minutes must be non-negative.")
    if not 0 <= options.positive_threshold <= 1:
        raise ValueError("--positive-threshold must be between 0 and 1.")
    if options.c <= 0:
        raise ValueError("--c must be greater than 0.")
    if options.split_strategy not in {"training_csv", "forecast-issue-blocked"}:
        raise ValueError("--split-strategy must be either 'training_csv' or 'forecast-issue-blocked'.")
    validate_blocked_fractions(options.blocked_val_fraction, options.blocked_test_fraction)


def forecast_weather_records_by_camera(conn: sqlite3.Connection) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, camera_id, valid_at_utc, fetched_at_utc, forecast_lead_hours, is_forecast, variables_json
        FROM weather_record
        ORDER BY camera_id, valid_at_utc, fetched_at_utc, id
        """
    ).fetchall()
    exact_keys: set[tuple[str, str, str, str]] = set()
    deduped_by_issue: dict[tuple[str, str, str], dict[str, Any]] = {}
    exact_duplicate_count = 0
    superseded_same_issue_count = 0
    non_forecast_count = 0
    invalid_json_count = 0
    for row in rows:
        variables_json = row["variables_json"] or "{}"
        exact_key = (row["camera_id"], row["valid_at_utc"], row["fetched_at_utc"], variables_json)
        if exact_key in exact_keys:
            exact_duplicate_count += 1
            continue
        exact_keys.add(exact_key)
        try:
            variables = json.loads(variables_json)
        except json.JSONDecodeError:
            invalid_json_count += 1
            continue
        valid_at = parse_datetime(row["valid_at_utc"])
        fetched_at = parse_datetime(row["fetched_at_utc"])
        lead_hours = row["forecast_lead_hours"]
        if lead_hours is None:
            lead_hours = (valid_at - fetched_at).total_seconds() / 3600.0
        is_forecast = bool(row["is_forecast"]) if row["is_forecast"] is not None else lead_hours > 0
        if not is_forecast or float(lead_hours) <= 0:
            non_forecast_count += 1
            continue
        issue_key = (row["camera_id"], row["valid_at_utc"], row["fetched_at_utc"])
        if issue_key in deduped_by_issue:
            superseded_same_issue_count += 1
        deduped_by_issue[issue_key] = {
            "id": int(row["id"]),
            "camera_id": row["camera_id"],
            "valid_at_utc": row["valid_at_utc"],
            "valid_at": valid_at,
            "fetched_at_utc": row["fetched_at_utc"],
            "fetched_at": fetched_at,
            "forecast_lead_hours": float(lead_hours),
            "variables": variables,
        }

    by_camera: dict[str, list[dict[str, Any]]] = {}
    for item in deduped_by_issue.values():
        by_camera.setdefault(item["camera_id"], []).append(item)
    for items in by_camera.values():
        items.sort(key=lambda item: (item["valid_at"], item["fetched_at"]))

    return by_camera, {
        "raw_weather_records": len(rows),
        "forecast_records_after_dedup": sum(len(items) for items in by_camera.values()),
        "exact_duplicate_records_excluded": exact_duplicate_count,
        "superseded_same_issue_records_excluded": superseded_same_issue_count,
        "non_forecast_records_excluded": non_forecast_count,
        "invalid_json_records_excluded": invalid_json_count,
        "camera_counts": {camera_id: len(items) for camera_id, items in sorted(by_camera.items())},
    }


def build_forecast_weather_examples(
    rows: list[dict[str, str]],
    weather: dict[str, list[dict[str, Any]]],
    *,
    requested_features: tuple[str, ...],
    forecast_horizon_hours: float,
    horizon_tolerance_minutes: float,
    max_weather_age_minutes: float,
    require_all_features: bool,
    positive_label: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    examples = []
    skipped = {
        "no_forecast_for_camera": 0,
        "no_forecast_valid_near_capture": 0,
        "no_forecast_at_requested_horizon": 0,
        "missing_required_features": 0,
        "no_numeric_features": 0,
    }
    required_features = set(requested_features)
    for row in rows:
        camera_weather = weather.get(row["camera_id"], [])
        if not camera_weather:
            skipped["no_forecast_for_camera"] += 1
            continue
        captured_at = parse_datetime(row["captured_at_utc"])
        candidates = []
        for record in camera_weather:
            weather_age_minutes = abs((captured_at - record["valid_at"]).total_seconds()) / 60.0
            if weather_age_minutes > max_weather_age_minutes:
                continue
            horizon_delta_minutes = abs((record["forecast_lead_hours"] - forecast_horizon_hours) * 60.0)
            if horizon_delta_minutes > horizon_tolerance_minutes:
                continue
            if record["fetched_at"] > captured_at:
                continue
            candidates.append((weather_age_minutes, horizon_delta_minutes, -record["fetched_at"].timestamp(), record))
        if not candidates:
            nearby = [
                record
                for record in camera_weather
                if abs((captured_at - record["valid_at"]).total_seconds()) / 60.0 <= max_weather_age_minutes
            ]
            if nearby:
                skipped["no_forecast_at_requested_horizon"] += 1
            else:
                skipped["no_forecast_valid_near_capture"] += 1
            continue
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        match = candidates[0][3]
        features = numeric_weather_features(match["variables"], requested_features)
        if require_all_features and required_features:
            missing = required_features - set(features)
            if missing:
                skipped["missing_required_features"] += 1
                continue
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
                "weather_fetched_at_utc": match["fetched_at_utc"],
                "forecast_lead_hours": match["forecast_lead_hours"],
                "forecast_horizon_delta_minutes": abs((match["forecast_lead_hours"] - forecast_horizon_hours) * 60.0),
                "weather_age_minutes": abs((captured_at - match["valid_at"]).total_seconds()) / 60.0,
                "forecast_group": forecast_group_key(row["camera_id"], match["valid_at_utc"], match["fetched_at_utc"]),
                "split": row["split"],
                "label": row["label"],
                "target": int(row["label"] == positive_label),
                "features": features,
            }
        )
    return examples, {key: value for key, value in skipped.items() if value}


def assign_forecast_issue_blocked_splits(
    examples: list[dict[str, Any]],
    *,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, Any]:
    train_fraction = 1.0 - val_fraction - test_fraction
    groups = grouped_forecast_examples(examples)
    group_rows = []
    for key, group_examples in groups.items():
        first = min(group_examples, key=lambda item: (item["captured_at_utc"], item["camera_id"], item["capture_id"]))
        group_rows.append(
            {
                "forecast_group": key,
                "first_captured_at_utc": first["captured_at_utc"],
                "camera_id": first["camera_id"],
                "target": int(any(example["target"] for example in group_examples)),
                "row_count": len(group_examples),
            }
        )

    split_by_group: dict[str, str] = {}
    for target in (0, 1):
        target_groups = [group for group in group_rows if group["target"] == target]
        target_groups.sort(key=lambda item: (item["first_captured_at_utc"], item["camera_id"], item["forecast_group"]))
        assigned = chronological_split_names(
            len(target_groups),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        for group, split in zip(target_groups, assigned):
            split_by_group[group["forecast_group"]] = split

    for example in examples:
        example["original_split"] = example["split"]
        example["split"] = split_by_group[example["forecast_group"]]

    return {
        "group_count": len(group_rows),
        "row_split_counts": split_counts(examples),
        "positive_group_count": sum(1 for group in group_rows if group["target"]),
        "non_positive_group_count": sum(1 for group in group_rows if not group["target"]),
    }


def collect_forecast_weather_predictions(
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
                "weather_fetched_at_utc": example["weather_fetched_at_utc"],
                "forecast_lead_hours": example["forecast_lead_hours"],
                "forecast_horizon_delta_minutes": example["forecast_horizon_delta_minutes"],
                "weather_age_minutes": example["weather_age_minutes"],
                "forecast_group": example["forecast_group"],
                "true_label": example["label"],
                "binary_true_label": binary_true_label,
                "pred_label": pred_label,
                "positive_probability": float(probability),
                "correct": int(binary_true_label == pred_label),
            }
        )
    return rows


def forecast_weather_report(predictions: list[dict[str, Any]], *, positive_label: str) -> dict[str, Any]:
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


def forecast_group_leakage(examples: list[dict[str, Any]]) -> dict[str, Any]:
    groups = grouped_forecast_examples(examples)
    split_sets = {key: {example["split"] for example in group} for key, group in groups.items()}
    train_groups = {key for key, splits in split_sets.items() if "train" in splits}
    leakage_rows = {
        split: sum(
            1
            for example in examples
            if example["split"] == split and example["forecast_group"] in train_groups
        )
        for split in ("val", "test")
    }
    leaked_groups = {key: sorted(splits) for key, splits in split_sets.items() if len(splits) > 1}
    return {
        "unique_forecast_groups": len(groups),
        "groups_spanning_multiple_splits": len(leaked_groups),
        "rows_sharing_forecast_group_with_train": leakage_rows,
        "is_blocked": len(leaked_groups) == 0,
    }


def matched_sanity(examples: list[dict[str, Any]], feature_names: list[str]) -> dict[str, Any]:
    missing_by_feature = {feature: 0 for feature in feature_names}
    for example in examples:
        for feature in feature_names:
            if feature not in example["features"]:
                missing_by_feature[feature] += 1
    lead_hours = [float(example["forecast_lead_hours"]) for example in examples]
    return {
        "matched_rows": len(examples),
        "required_feature_count": len(feature_names),
        "missing_by_feature": {key: value for key, value in missing_by_feature.items() if value},
        "min_forecast_lead_hours": min(lead_hours) if lead_hours else None,
        "max_forecast_lead_hours": max(lead_hours) if lead_hours else None,
    }


def grouped_forecast_examples(examples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        groups.setdefault(example["forecast_group"], []).append(example)
    return groups


def forecast_group_key(camera_id: str, weather_valid_at_utc: str, weather_fetched_at_utc: str) -> str:
    return f"{camera_id}|{weather_valid_at_utc}|{weather_fetched_at_utc}"


def write_forecast_weather_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "split",
        "capture_id",
        "camera_id",
        "captured_at_utc",
        "weather_valid_at_utc",
        "weather_fetched_at_utc",
        "forecast_lead_hours",
        "forecast_horizon_delta_minutes",
        "weather_age_minutes",
        "forecast_group",
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


def write_forecast_coefficients(rows: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "coefficient", "abs_coefficient"])
        writer.writeheader()
        writer.writerows(rows)
