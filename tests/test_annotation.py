from pathlib import Path

from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.annotation import (
    annotation_stats,
    delete_annotation,
    next_unannotated_frame,
    save_annotation,
    task_labels,
)
from enviro_webcam_ml.config import load_config


def test_task_labels_use_config_order() -> None:
    config = load_config(Path("configs/mount_tam.yaml"))

    assert task_labels(config, "marine_layer_detection") == [
        "clouds_below_peak",
        "no_clouds_below_peak",
        "peak_obscured",
        "uncertain",
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
