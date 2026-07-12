from __future__ import annotations

import json
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enviro_webcam_ml.paired_events import POSITIVE_EVENT_LABEL
from enviro_webcam_ml.paired_weather_lasso import (
    assign_paired_weather_blocked_splits,
    build_paired_weather_examples,
    collect_paired_weather_predictions,
    paired_weather_group_leakage,
    paired_weather_report,
    read_paired_event_rows,
    validate_paired_split_strategy,
    write_coefficients,
    write_paired_weather_predictions,
)
from enviro_webcam_ml.weather_lasso import (
    assign_weather_hour_blocked_splits,
    build_weather_examples,
    collect_weather_predictions,
    feature_matrix,
    inferred_feature_names,
    read_weather_training_rows,
    split_counts,
    validate_blocked_fractions,
    validate_split_strategy,
    weather_binary_report,
    weather_group_leakage,
    weather_records_by_camera,
    write_weather_predictions,
)


SUPPORTED_WEATHER_CLASSIFIERS = ("ridge_logistic", "random_forest", "hist_gradient_boosting")


@dataclass(frozen=True)
class WeatherClassifierOptions:
    training_csv: Path
    output_dir: Path
    positive_label: str
    model_kind: str
    features: tuple[str, ...] = ()
    max_weather_age_minutes: float = 90.0
    c: float = 1.0
    positive_threshold: float = 0.5
    class_weight: str | None = None
    split_strategy: str = "weather-hour-blocked"
    blocked_val_fraction: float = 0.15
    blocked_test_fraction: float = 0.15
    random_state: int = 42


@dataclass(frozen=True)
class PairedWeatherClassifierOptions:
    paired_events_csv: Path
    output_dir: Path
    camera_ids: tuple[str, str]
    model_kind: str
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


def train_weather_classifier(
    conn: sqlite3.Connection,
    options: WeatherClassifierOptions,
) -> dict[str, Any]:
    validate_weather_classifier_kind(options.model_kind)
    validate_split_strategy(options.split_strategy)
    validate_blocked_fractions(options.blocked_val_fraction, options.blocked_test_fraction)
    rows = read_weather_training_rows(options.training_csv)
    if not rows:
        raise ValueError(f"No rows found in training CSV: {options.training_csv}")
    if options.positive_label not in {row["label"] for row in rows}:
        raise ValueError(f"positive_label={options.positive_label!r} is not present in the training CSV.")
    examples, skipped = build_weather_examples(
        rows,
        weather_records_by_camera(conn),
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
        split_summary = assign_weather_hour_blocked_splits(
            examples,
            val_fraction=options.blocked_val_fraction,
            test_fraction=options.blocked_test_fraction,
        )
    else:
        split_summary = None
    return train_weather_classifier_from_examples(
        examples,
        output_dir=options.output_dir,
        model_kind=options.model_kind,
        positive_label=options.positive_label,
        positive_threshold=options.positive_threshold,
        c=options.c,
        class_weight=options.class_weight,
        random_state=options.random_state,
        split_strategy=options.split_strategy,
        feature_names=list(options.features or inferred_feature_names(examples)),
        max_weather_age_minutes=options.max_weather_age_minutes,
        skipped=skipped,
        split_summary=split_summary,
        leakage_summary=weather_group_leakage(examples),
        extra_metadata={"training_csv": str(options.training_csv), "event_scope": "single_image"},
        predictions_collector=collect_weather_predictions,
        report_builder=weather_binary_report,
        predictions_writer=write_weather_predictions,
    )


def train_paired_weather_classifier(
    conn: sqlite3.Connection,
    options: PairedWeatherClassifierOptions,
) -> dict[str, Any]:
    validate_weather_classifier_kind(options.model_kind)
    validate_paired_split_strategy(options.split_strategy)
    validate_blocked_fractions(options.val_fraction, options.test_fraction)
    if len(options.camera_ids) != 2:
        raise ValueError("Paired weather classifier requires exactly two camera IDs.")
    rows = read_paired_event_rows(options.paired_events_csv)
    if not rows:
        raise ValueError(f"No paired event rows found in {options.paired_events_csv}")
    examples, skipped = build_paired_weather_examples(
        rows,
        weather_records_by_camera(conn),
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
        from enviro_webcam_ml.paired_weather_lasso import assign_paired_chronological_splits

        split_summary = assign_paired_chronological_splits(
            examples,
            train_fraction=options.train_fraction,
            val_fraction=options.val_fraction,
        )
    return train_weather_classifier_from_examples(
        examples,
        output_dir=options.output_dir,
        model_kind=options.model_kind,
        positive_label=options.positive_label,
        positive_threshold=options.positive_threshold,
        c=options.c,
        class_weight=options.class_weight,
        random_state=options.random_state,
        split_strategy=options.split_strategy,
        feature_names=list(options.features or inferred_feature_names(examples)),
        max_weather_age_minutes=options.max_weather_age_minutes,
        skipped=skipped,
        split_summary=split_summary,
        leakage_summary=paired_weather_group_leakage(examples),
        extra_metadata={
            "paired_events_csv": str(options.paired_events_csv),
            "event_scope": "paired_event",
            "camera_ids": list(options.camera_ids),
        },
        predictions_collector=collect_paired_weather_predictions,
        report_builder=paired_weather_report,
        predictions_writer=write_paired_weather_predictions,
    )


def train_weather_classifier_from_examples(
    examples: list[dict[str, Any]],
    *,
    output_dir: Path,
    model_kind: str,
    positive_label: str,
    positive_threshold: float,
    c: float,
    class_weight: str | None,
    random_state: int,
    split_strategy: str,
    feature_names: list[str],
    max_weather_age_minutes: float,
    skipped: dict[str, int],
    split_summary: dict[str, Any] | None,
    leakage_summary: dict[str, Any],
    extra_metadata: dict[str, Any],
    predictions_collector,
    report_builder,
    predictions_writer,
) -> dict[str, Any]:
    if not 0 <= positive_threshold <= 1:
        raise ValueError("positive_threshold must be between 0 and 1.")
    if not feature_names:
        raise ValueError("No numeric weather features were found.")
    train_examples = [example for example in examples if example["split"] == "train"]
    if not train_examples:
        raise ValueError("No train rows with matched weather records.")
    if len({example["target"] for example in train_examples}) < 2:
        raise ValueError("Train split needs both positive and non-positive weather examples.")
    pipeline = build_weather_pipeline(
        model_kind,
        c=c,
        class_weight=class_weight,
        random_state=random_state,
    )
    pipeline.fit(feature_matrix(train_examples, feature_names), [example["target"] for example in train_examples])
    prediction_rows = predictions_collector(
        pipeline,
        examples,
        feature_names,
        positive_label=positive_label,
        positive_threshold=positive_threshold,
    )
    detailed_metrics = {
        split: report_builder(
            [row for row in prediction_rows if row["split"] == split],
            positive_label=positive_label,
        )
        for split in ("train", "val", "test")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pkl"
    metadata_path = output_dir / "metadata.json"
    predictions_path = output_dir / "predictions.csv"
    importances_path = output_dir / "feature_importances.csv"
    importances = weather_feature_importance_rows(pipeline, feature_names)
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "pipeline": pipeline,
                "feature_names": feature_names,
                "positive_label": positive_label,
                "positive_threshold": positive_threshold,
                "model_kind": model_kind,
            },
            f,
        )
    summary = {
        **extra_metadata,
        "model_name": f"weather_{model_kind}",
        "model_type": "weather_classifier",
        "model_kind": model_kind,
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "feature_importances_path": str(importances_path),
        "positive_label": positive_label,
        "positive_threshold": positive_threshold,
        "max_weather_age_minutes": max_weather_age_minutes,
        "c": c,
        "class_weight": normalize_class_weight(class_weight) or "none",
        "split_strategy": split_strategy,
        "features": feature_names,
        "feature_importances": importances,
        "nonzero_coefficients": [
            row for row in importances
            if row.get("coefficient") not in (None, 0.0)
        ],
        "split_counts": split_counts(examples),
        "matched_rows": len(examples),
        "skipped": skipped,
        "blocked_split_summary": split_summary,
        "weather_group_leakage": leakage_summary,
        "detailed_metrics": detailed_metrics,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    predictions_writer(prediction_rows, predictions_path)
    write_coefficients(importances, importances_path)
    return summary


def build_weather_pipeline(model_kind: str, *, c: float, class_weight: str | None, random_state: int):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError(
            "Weather classifier training requires scikit-learn. Install with: "
            'python -m pip install -e ".[dev,train]"'
        ) from exc
    class_weight = normalize_class_weight(class_weight)
    if class_weight not in (None, "balanced"):
        raise ValueError("--class-weight must be either 'none' or 'balanced'.")
    if model_kind == "ridge_logistic":
        if c <= 0:
            raise ValueError("--c must be greater than 0.")
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        C=c,
                        class_weight=class_weight,
                        random_state=random_state,
                        max_iter=1000,
                    ),
                ),
            ]
        )
    if model_kind == "random_forest":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=5,
                        min_samples_leaf=3,
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if model_kind == "hist_gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=200,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    validate_weather_classifier_kind(model_kind)
    raise AssertionError("unreachable")


def weather_feature_importance_rows(pipeline, feature_names: list[str]) -> list[dict[str, Any]]:
    model = pipeline.named_steps["model"]
    rows = []
    if hasattr(model, "coef_"):
        for feature, coefficient in zip(feature_names, model.coef_[0]):
            value = float(coefficient)
            rows.append({"feature": feature, "coefficient": value, "abs_coefficient": abs(value)})
        return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)
    if hasattr(model, "feature_importances_"):
        for feature, importance in zip(feature_names, model.feature_importances_):
            value = float(importance)
            rows.append({"feature": feature, "coefficient": value, "abs_coefficient": abs(value)})
        return sorted(rows, key=lambda row: row["abs_coefficient"], reverse=True)
    return []


def validate_weather_classifier_kind(model_kind: str) -> None:
    if model_kind not in SUPPORTED_WEATHER_CLASSIFIERS:
        raise ValueError(
            f"Unsupported weather classifier {model_kind!r}; "
            f"choose one of {', '.join(SUPPORTED_WEATHER_CLASSIFIERS)}."
        )


def normalize_class_weight(value: str | None) -> str | None:
    return None if value in (None, "", "none") else value
