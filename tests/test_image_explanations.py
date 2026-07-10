from pathlib import Path

from PIL import Image

from enviro_webcam_ml.image_explanations import (
    explanation_filename,
    read_prediction_rows,
    select_prediction_rows,
)


def test_read_prediction_rows_keeps_only_existing_images(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8), "blue").save(image_path)
    missing_path = tmp_path / "missing.jpg"
    predictions = tmp_path / "predictions.csv"
    predictions.write_text(
        "split,capture_id,camera_id,captured_at_utc,true_label,pred_label,confidence,correct,image_path\n"
        f"test,1,camera_a,2026-01-01T00:00:00Z,yes,yes,0.9,1,{image_path}\n"
        f"test,2,camera_a,2026-01-01T00:05:00Z,no,no,0.8,1,{missing_path}\n",
        encoding="utf-8",
    )

    rows = read_prediction_rows(predictions)

    assert len(rows) == 1
    assert rows[0]["capture_id"] == "1"


def test_select_prediction_rows_mixed_prioritizes_errors_then_low_confidence() -> None:
    rows = [
        {"split": "test", "capture_id": "easy", "confidence": "0.99", "correct": "1"},
        {"split": "test", "capture_id": "borderline", "confidence": "0.51", "correct": "1"},
        {"split": "test", "capture_id": "mistake_low", "confidence": "0.6", "correct": "0"},
        {"split": "test", "capture_id": "mistake_high", "confidence": "0.95", "correct": "0"},
        {"split": "val", "capture_id": "wrong_split", "confidence": "0.1", "correct": "0"},
    ]

    selected = select_prediction_rows(rows, split="test", selection="mixed", max_images=3)

    assert [row["capture_id"] for row in selected] == [
        "mistake_high",
        "mistake_low",
        "borderline",
    ]


def test_select_prediction_rows_can_filter_by_true_and_pred_labels() -> None:
    rows = [
        {
            "split": "test",
            "capture_id": "true_positive",
            "true_label": "clouds_below_peak",
            "pred_label": "clouds_below_peak",
            "confidence": "0.9",
            "correct": "1",
        },
        {
            "split": "test",
            "capture_id": "false_negative",
            "true_label": "clouds_below_peak",
            "pred_label": "no_clouds_below_peak",
            "confidence": "0.6",
            "correct": "0",
        },
        {
            "split": "test",
            "capture_id": "negative",
            "true_label": "no_clouds_below_peak",
            "pred_label": "no_clouds_below_peak",
            "confidence": "0.7",
            "correct": "1",
        },
    ]

    selected = select_prediction_rows(
        rows,
        split="test",
        selection="mixed",
        max_images=10,
        true_labels=("clouds_below_peak",),
        pred_labels=("clouds_below_peak",),
    )

    assert [row["capture_id"] for row in selected] == ["true_positive"]


def test_explanation_filename_is_safe() -> None:
    filename = explanation_filename(
        1,
        {
            "split": "test",
            "camera_id": "camera/a",
            "capture_id": "abc 123",
            "true_label": "clouds below peak",
            "pred_label": "no/clouds",
            "image_path": "/tmp/frame.jpg",
        },
    )

    assert filename == "001_test_camera_a_abc_123_true-clouds_below_peak_pred-no_clouds.jpg"
