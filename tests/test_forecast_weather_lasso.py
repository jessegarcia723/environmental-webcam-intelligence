import csv
from pathlib import Path

import pytest

from enviro_webcam_ml import db
from enviro_webcam_ml.forecast_weather_lasso import (
    ForecastWeatherLassoOptions,
    train_forecast_weather_lasso,
)


pytest.importorskip("sklearn")


def test_train_forecast_weather_lasso_uses_prior_forecast_issue_and_dedupes(tmp_path: Path) -> None:
    db_path = tmp_path / "forecast.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "forecast_weather_lasso"
    write_training_csv(training_csv)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_forecast_weather(conn)
        # Exact duplicate from the same fetch should be excluded by forecast sanity logic.
        db.insert_weather_records(
            conn,
            provider="test",
            camera_id="camera",
            fetched_at_utc="2026-07-07T21:00:00+00:00",
            records=[
                {
                    "valid_at_utc": "2026-07-08T00:00:00+00:00",
                    "variables": weather_variables(True),
                }
            ],
        )

        summary = train_forecast_weather_lasso(
            conn,
            ForecastWeatherLassoOptions(
                training_csv=training_csv,
                output_dir=output_dir,
                positive_label="clouds_below_peak",
                features=("cloud_cover_low", "relative_humidity_2m"),
                forecast_horizon_hours=3.0,
                c=10.0,
            ),
        )

    assert Path(summary["model_path"]).exists()
    assert Path(summary["metadata_path"]).exists()
    assert Path(summary["predictions_path"]).exists()
    assert Path(summary["coefficients_path"]).exists()
    assert summary["matched_rows"] == 8
    assert summary["weather_sanity"]["exact_duplicate_records_excluded"] == 1
    assert summary["weather_sanity"]["forecast_records_after_dedup"] == 8
    assert summary["matched_sanity"]["missing_by_feature"] == {}
    assert summary["forecast_group_leakage"]["is_blocked"] is True
    assert summary["detailed_metrics"]["test"]["binary"]["count"] == 2

    predictions = list(csv.DictReader(Path(summary["predictions_path"]).open(encoding="utf-8")))
    assert {"weather_fetched_at_utc", "forecast_lead_hours", "forecast_group"} <= set(predictions[0])
    assert all(float(row["forecast_lead_hours"]) == 3.0 for row in predictions)


def test_train_forecast_weather_lasso_skips_missing_required_features(tmp_path: Path) -> None:
    db_path = tmp_path / "forecast.sqlite3"
    training_csv = tmp_path / "training.csv"
    output_dir = tmp_path / "forecast_weather_lasso"
    write_training_csv(training_csv)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        insert_forecast_weather(conn, include_humidity=False)
        with pytest.raises(ValueError, match="No training rows could be matched"):
            train_forecast_weather_lasso(
                conn,
                ForecastWeatherLassoOptions(
                    training_csv=training_csv,
                    output_dir=output_dir,
                    positive_label="clouds_below_peak",
                    features=("cloud_cover_low", "relative_humidity_2m"),
                    forecast_horizon_hours=3.0,
                    require_all_features=True,
                ),
            )


def test_init_db_migrates_weather_forecast_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    conn = __import__("sqlite3").connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE weather_record (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              provider TEXT NOT NULL,
              camera_id TEXT NOT NULL,
              valid_at_utc TEXT NOT NULL,
              fetched_at_utc TEXT NOT NULL,
              variables_json TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO weather_record (
              provider, camera_id, valid_at_utc, fetched_at_utc, variables_json
            )
            VALUES ('test', 'camera', '2026-07-08T03:00:00+00:00', '2026-07-08T00:00:00+00:00', '{}')
            """
        )
        conn.commit()
    finally:
        conn.close()

    db.init_db(db_path)

    with db.connect(db_path) as migrated:
        row = migrated.execute(
            "SELECT forecast_lead_hours, is_forecast FROM weather_record"
        ).fetchone()
    assert row["forecast_lead_hours"] == 3.0
    assert row["is_forecast"] == 1


def write_training_csv(path: Path) -> None:
    labels = ["clouds_below_peak"] * 4 + ["no_clouds_below_peak"] * 4
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["capture_id", "camera_id", "captured_at_utc", "split", "label"],
        )
        writer.writeheader()
        for hour, label in enumerate(labels):
            writer.writerow(
                {
                    "capture_id": str(hour + 1),
                    "camera_id": "camera",
                    "captured_at_utc": f"2026-07-08T{hour:02d}:05:00+00:00",
                    "split": "train" if hour in (0, 1, 4, 5) else "test",
                    "label": label,
                }
            )


def insert_forecast_weather(conn, *, include_humidity: bool = True) -> None:
    for hour in range(8):
        positive_weather = hour < 4
        variables = weather_variables(positive_weather)
        if not include_humidity:
            variables.pop("relative_humidity_2m")
        db.insert_weather_records(
            conn,
            provider="test",
            camera_id="camera",
            fetched_at_utc=f"2026-07-07T{21 + hour:02d}:00:00+00:00"
            if hour < 3
            else f"2026-07-08T{hour - 3:02d}:00:00+00:00",
            records=[
                {
                    "valid_at_utc": f"2026-07-08T{hour:02d}:00:00+00:00",
                    "variables": variables,
                }
            ],
        )


def weather_variables(positive_weather: bool) -> dict[str, float]:
    return {
        "cloud_cover_low": 95.0 if positive_weather else 5.0,
        "relative_humidity_2m": 85.0 if positive_weather else 30.0,
    }
