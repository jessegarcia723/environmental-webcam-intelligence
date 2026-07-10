from pathlib import Path

from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.annotation import (
    adjudication_report,
    annotation_stats,
    delete_annotation,
    favorite_rows,
    image_path_for_capture,
    load_model_predictions,
    next_unannotated_frame,
    next_adjudication_case,
    pacific_time_label,
    save_annotation,
    save_adjudication,
    save_favorite,
    task_labels,
)
from enviro_webcam_ml.config import load_config


def test_task_labels_use_config_order() -> None:
    config = load_config(Path("configs/mount_tam.yaml"))

    assert task_labels(config, "marine_layer_detection") == [
        "clouds_below_peak",
        "no_clouds_below_peak",
        "peak_obscured",
        "below_peak_height_far_from_peak",
        "night_unusable",
        "camera_artifact",
    ]


def test_next_unannotated_frame_skips_existing_annotator_label(tmp_path: Path) -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    db_path = tmp_path / "annotations.sqlite3"
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8), color=(120, 130, 140)).save(image_path)

    db.init_db(db_path)
    with db.connect(db_path) as conn:
        db.register_config(conn, config)
        first_capture_id = db.insert_capture(
            conn,
            camera_id="mount_tam_east_peak",
            pose_version="initial",
            captured_at_utc="2026-07-08T12:00:00+00:00",
            requested_url="https://example.test/1.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=100,
            sha256="abc",
            error=None,
        )
        second_capture_id = db.insert_capture(
            conn,
            camera_id="mount_tam_east_peak",
            pose_version="initial",
            captured_at_utc="2026-07-08T12:05:00+00:00",
            requested_url="https://example.test/2.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=100,
            sha256="def",
            error=None,
        )
        db.insert_image_asset(
            conn,
            capture_id=first_capture_id,
            path=image_path,
            sha256="abc",
            width=8,
            height=8,
        )
        db.insert_image_asset(
            conn,
            capture_id=second_capture_id,
            path=image_path,
            sha256="def",
            width=8,
            height=8,
        )

        first = next_unannotated_frame(
            conn,
            task_id="marine_layer_detection",
            annotator="jesse",
        )
        assert first is not None
        assert first["capture_id"] == first_capture_id
        assert first["captured_at_pacific"] == "2026-07-08 05:00:00 AM PDT"

        save_annotation(
            conn,
            capture_id=first_capture_id,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="jesse",
        )

        second = next_unannotated_frame(
            conn,
            task_id="marine_layer_detection",
            annotator="jesse",
        )
        assert second is not None
        assert second["capture_id"] == second_capture_id

        other_annotator_first = next_unannotated_frame(
            conn,
            task_id="marine_layer_detection",
            annotator="partner",
        )
        assert other_annotator_first is not None
        assert other_annotator_first["capture_id"] == first_capture_id


def test_image_path_for_capture_remaps_old_data_root(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite3"
    data_dir = tmp_path / "synced" / "data"
    image_path = data_dir / "raw" / "mount_tam_east_peak" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(120, 130, 140)).save(image_path)
    old_path = "/Users/old/environmental-webcam-intelligence/data/raw/mount_tam_east_peak/frame.jpg"

    db.init_db(db_path)
    with db.connect(db_path) as conn:
        capture_id = db.insert_capture(
            conn,
            camera_id="mount_tam_east_peak",
            pose_version="initial",
            captured_at_utc="2026-07-08T12:00:00+00:00",
            requested_url="https://example.test/1.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=100,
            sha256="abc",
            error=None,
        )
        db.insert_image_asset(
            conn,
            capture_id=capture_id,
            path=Path(old_path),
            sha256="abc",
            width=8,
            height=8,
        )

        resolved = image_path_for_capture(conn, capture_id, data_dir=data_dir)

    assert resolved == str(image_path.resolve())


def test_save_annotation_updates_same_annotator_row(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite3"
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        capture_id = db.insert_capture(
            conn,
            camera_id="camera",
            pose_version="initial",
            captured_at_utc="2026-07-08T12:00:00+00:00",
            requested_url="https://example.test/1.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=100,
            sha256="abc",
            error=None,
        )
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="jesse",
        )
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label="no_clouds_below_peak",
            annotator="jesse",
        )

        rows = conn.execute("SELECT label FROM annotation").fetchall()
        stats = annotation_stats(conn, task_id="marine_layer_detection")

    assert [row["label"] for row in rows] == ["no_clouds_below_peak"]
    assert stats["totals"] == [{"annotator": "jesse", "count": 1}]
    assert stats["favorite_count"] == 0


def test_save_favorite_is_separate_from_training_labels(tmp_path: Path) -> None:
    db_path = tmp_path / "favorites.sqlite3"
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8), color=(120, 130, 140)).save(image_path)
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        capture_id = db.insert_capture(
            conn,
            camera_id="camera",
            pose_version="initial",
            captured_at_utc="2026-07-08T12:00:00+00:00",
            requested_url="https://example.test/1.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=100,
            sha256="abc",
            error=None,
        )
        db.insert_image_asset(
            conn,
            capture_id=capture_id,
            path=image_path,
            sha256="abc",
            width=8,
            height=8,
        )
        save_favorite(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            annotator="jesse",
            notes="pretty cloud deck",
        )
        save_favorite(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            annotator="jesse",
            notes="updated note",
        )
        rows = favorite_rows(conn, task_id="marine_layer_detection")
        annotations = conn.execute("SELECT COUNT(*) AS count FROM annotation").fetchone()["count"]
        stats = annotation_stats(conn, task_id="marine_layer_detection")

    assert annotations == 0
    assert stats["favorite_count"] == 1
    assert len(rows) == 1
    assert rows[0]["notes"] == "updated note"
    assert rows[0]["captured_at_pacific"] == "2026-07-08 05:00:00 AM PDT"


def test_pacific_time_label_handles_winter_standard_time() -> None:
    assert pacific_time_label("2026-01-08T12:00:00+00:00") == "2026-01-08 04:00:00 AM PST"


def test_delete_annotation_removes_only_matching_annotator(tmp_path: Path) -> None:
    db_path = tmp_path / "annotations.sqlite3"
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        capture_id = db.insert_capture(
            conn,
            camera_id="camera",
            pose_version="initial",
            captured_at_utc="2026-07-08T12:00:00+00:00",
            requested_url="https://example.test/1.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=100,
            sha256="abc",
            error=None,
        )
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="jesse",
        )
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label="no_clouds_below_peak",
            annotator="partner",
        )

        deleted = delete_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            annotator="jesse",
        )
        rows = conn.execute(
            "SELECT annotator, label FROM annotation ORDER BY annotator"
        ).fetchall()

    assert deleted is True
    assert [dict(row) for row in rows] == [
        {"annotator": "partner", "label": "no_clouds_below_peak"}
    ]


def test_adjudication_case_report_and_prediction_lookup(tmp_path: Path) -> None:
    db_path = tmp_path / "adjudication.sqlite3"
    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (8, 8), color=(120, 130, 140)).save(image_path)
    predictions_csv = tmp_path / "predictions.csv"
    predictions_csv.write_text(
        "split,capture_id,camera_id,captured_at_utc,true_label,pred_label,confidence,correct,image_path\n"
        f"test,1,camera,2026-07-08T12:00:00+00:00,clouds_below_peak,no_clouds_below_peak,0.81,0,{image_path}\n",
        encoding="utf-8",
    )

    db.init_db(db_path)
    predictions = load_model_predictions(predictions_csv)
    with db.connect(db_path) as conn:
        capture_id = db.insert_capture(
            conn,
            camera_id="camera",
            pose_version="initial",
            captured_at_utc="2026-07-08T12:00:00+00:00",
            requested_url="https://example.test/1.jpg",
            http_status=200,
            content_type="image/jpeg",
            byte_count=100,
            sha256="abc",
            error=None,
        )
        db.insert_image_asset(
            conn,
            capture_id=capture_id,
            path=image_path,
            sha256="abc",
            width=8,
            height=8,
        )
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="jesse",
        )
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label="no_clouds_below_peak",
            annotator="partner",
        )
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label="peak_obscured_uncertain",
            annotator="old_test",
        )

        case = next_adjudication_case(
            conn,
            task_id="marine_layer_detection",
            predictions=predictions,
            annotators=("jesse", "partner"),
        )
        assert case is not None
        assert case["agreement"] is False
        assert case["model_prediction"]["pred_label"] == "no_clouds_below_peak"
        assert case["captured_at_pacific"] == "2026-07-08 05:00:00 AM PDT"

        save_adjudication(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            final_label="clouds_below_peak",
            adjudicator="joint",
            model_label=case["model_prediction"]["pred_label"],
            model_confidence=case["model_prediction"]["confidence"],
        )
        report = adjudication_report(conn, task_id="marine_layer_detection")
        filtered_report = adjudication_report(
            conn,
            task_id="marine_layer_detection",
            annotators=("jesse", "partner"),
        )
        next_case = next_adjudication_case(
            conn,
            task_id="marine_layer_detection",
            predictions=predictions,
            annotators=("jesse", "partner"),
        )

    assert report["double_labeled"] == 1
    assert report["disagreements"] == 1
    assert filtered_report["disagreements"] == 1
    assert report["adjudicated"] == 1
    assert report["favorite_count"] == 0
    assert report["remaining_disagreements"] == 0
    assert report["final_label_counts"] == {"clouds_below_peak": 1}
    assert next_case is None
