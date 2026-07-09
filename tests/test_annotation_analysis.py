from pathlib import Path

from enviro_webcam_ml import db
from enviro_webcam_ml.annotation import save_annotation
from enviro_webcam_ml.annotation_analysis import (
    analyze_annotations,
    write_analysis_markdown,
    write_disagreements_csv,
)
from enviro_webcam_ml.config import load_config


def test_analyze_annotations_counts_agreement_disagreement_and_legacy_label(tmp_path: Path) -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    db_path = tmp_path / "analysis.sqlite3"
    db.init_db(db_path)

    with db.connect(db_path) as conn:
        db.register_config(conn, config)
        first_capture = insert_capture(conn, "2026-07-08T12:00:00+00:00")
        second_capture = insert_capture(conn, "2026-07-08T12:05:00+00:00")
        third_capture = insert_capture(conn, "2026-07-08T12:10:00+00:00")

        save_annotation(
            conn,
            capture_id=first_capture,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="jesse",
        )
        save_annotation(
            conn,
            capture_id=first_capture,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="partner",
        )
        save_annotation(
            conn,
            capture_id=second_capture,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="jesse",
        )
        save_annotation(
            conn,
            capture_id=second_capture,
            task_id="marine_layer_detection",
            label="no_clouds_below_peak",
            annotator="partner",
        )
        save_annotation(
            conn,
            capture_id=third_capture,
            task_id="marine_layer_detection",
            label="peak_obscured_uncertain",
            annotator="jesse",
        )

        analysis = analyze_annotations(
            conn,
            config=config,
            task_id="marine_layer_detection",
        )

    assert analysis["annotation_count"] == 5
    assert analysis["unique_capture_count"] == 3
    assert analysis["double_labeled_capture_count"] == 2
    assert analysis["legacy_labels"] == {"peak_obscured_uncertain": 1}
    assert len(analysis["disagreements"]) == 1
    assert analysis["disagreements"][0]["capture_id"] == second_capture
    assert len(analysis["pair_agreements"]) == 1
    assert analysis["pair_agreements"][0].overlap_count == 2
    assert analysis["pair_agreements"][0].agreement_count == 1


def test_write_analysis_outputs(tmp_path: Path) -> None:
    analysis = {
        "task_id": "marine_layer_detection",
        "configured_labels": ["clouds_below_peak", "no_clouds_below_peak"],
        "annotation_count": 2,
        "unique_capture_count": 1,
        "double_labeled_capture_count": 1,
        "total_by_annotator": {"jesse": 1, "partner": 1},
        "count_by_label": {"clouds_below_peak": 1, "no_clouds_below_peak": 1},
        "count_by_annotator_label": {},
        "legacy_labels": {},
        "pair_agreements": [],
        "disagreements": [
            {
                "capture_id": 1,
                "captured_at_utc": "2026-07-08T12:00:00+00:00",
                "camera_id": "mount_tam_east_peak",
                "image_path": "/tmp/frame.jpg",
                "annotations": {
                    "jesse": "clouds_below_peak",
                    "partner": "no_clouds_below_peak",
                },
            }
        ],
    }
    report = tmp_path / "report.md"
    disagreements = tmp_path / "disagreements.csv"

    write_analysis_markdown(analysis, report)
    write_disagreements_csv(analysis["disagreements"], disagreements)

    assert "Annotation analysis" in report.read_text(encoding="utf-8")
    csv_text = disagreements.read_text(encoding="utf-8")
    assert "capture_id" in csv_text
    assert "clouds_below_peak" in csv_text


def insert_capture(conn, captured_at_utc: str) -> int:
    return db.insert_capture(
        conn,
        camera_id="mount_tam_east_peak",
        pose_version="initial",
        captured_at_utc=captured_at_utc,
        requested_url="https://example.test/frame.jpg",
        http_status=200,
        content_type="image/jpeg",
        byte_count=100,
        sha256=captured_at_utc,
        error=None,
    )
