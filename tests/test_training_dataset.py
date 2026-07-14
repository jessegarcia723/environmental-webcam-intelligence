import csv
from pathlib import Path

from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.annotation import save_annotation
from enviro_webcam_ml.annotation import save_adjudication
from enviro_webcam_ml.config import load_config
from enviro_webcam_ml.training_dataset import (
    TrainingSetOptions,
    assign_stratified_chronological_splits,
    build_training_set,
    resolve_image_path,
    split_label_counts,
)


def test_resolve_image_path_remaps_old_data_root(tmp_path: Path) -> None:
    data_dir = tmp_path / "synced" / "data"
    image = data_dir / "raw" / "cam" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    old_path = "/Users/old/environmental-webcam-intelligence/data/raw/cam/frame.jpg"

    resolved = resolve_image_path(old_path, data_dir)

    assert resolved == image.resolve()
    assert resolved.exists()


def test_build_training_set_uses_only_agreed_current_non_excluded_labels(tmp_path: Path) -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    db_path = tmp_path / "training.sqlite3"
    output = tmp_path / "training.csv"
    image_path = tmp_path / "data" / "raw" / "mount_tam_east_peak" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(100, 120, 140)).save(image_path)

    db.init_db(db_path)
    with db.connect(db_path) as conn:
        db.register_config(conn, config)
        agreed = insert_capture_with_image(conn, image_path, "2026-07-08T12:00:00+00:00", "a")
        disagreement = insert_capture_with_image(conn, image_path, "2026-07-08T12:05:00+00:00", "b")
        excluded = insert_capture_with_image(conn, image_path, "2026-07-08T12:10:00+00:00", "c")
        legacy = insert_capture_with_image(conn, image_path, "2026-07-08T12:15:00+00:00", "d")
        single = insert_capture_with_image(conn, image_path, "2026-07-08T12:20:00+00:00", "e")

        add_two_labels(conn, agreed, "clouds_below_peak", "clouds_below_peak")
        add_two_labels(conn, disagreement, "clouds_below_peak", "no_clouds_below_peak")
        add_two_labels(conn, excluded, "night_unusable", "night_unusable")
        add_two_labels(conn, legacy, "peak_obscured_uncertain", "peak_obscured_uncertain")
        save_annotation(
            conn,
            capture_id=single,
            task_id="marine_layer_detection",
            label="clouds_below_peak",
            annotator="jesse",
        )

        summary = build_training_set(
            conn,
            config=config,
            options=TrainingSetOptions(
                task_id="marine_layer_detection",
                output_path=output,
                exclude_labels=config.task_excluded_training_labels("marine_layer_detection"),
            ),
        )

    csv_text = output.read_text(encoding="utf-8")
    assert summary["row_count"] == 1
    assert summary["label_counts"] == {"clouds_below_peak": 1}
    assert summary["skipped"] == {
        "disagreement": 1,
        "excluded_label": 1,
        "legacy_or_unknown_label": 1,
        "too_few_annotators": 1,
    }
    assert "clouds_below_peak" in csv_text
    assert "night_unusable" not in csv_text


def test_build_training_set_uses_adjudicated_disagreement_label(tmp_path: Path) -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    db_path = tmp_path / "adjudicated_training.sqlite3"
    output = tmp_path / "training.csv"
    image_path = tmp_path / "data" / "raw" / "mount_tam_east_peak" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(100, 120, 140)).save(image_path)

    db.init_db(db_path)
    with db.connect(db_path) as conn:
        db.register_config(conn, config)
        capture_id = insert_capture_with_image(conn, image_path, "2026-07-08T12:00:00+00:00", "a")
        add_two_labels(conn, capture_id, "clouds_below_peak", "no_clouds_below_peak")
        save_adjudication(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            final_label="clouds_below_peak",
            adjudicator="joint",
        )

        summary = build_training_set(
            conn,
            config=config,
            options=TrainingSetOptions(
                task_id="marine_layer_detection",
                output_path=output,
                exclude_labels=config.task_excluded_training_labels("marine_layer_detection"),
            ),
        )

    csv_text = output.read_text(encoding="utf-8")
    assert summary["row_count"] == 1
    assert summary["skipped"] == {}
    assert "clouds_below_peak" in csv_text
    assert "adjudicated" in csv_text


def test_build_training_set_respects_candidate_min_spacing(tmp_path: Path) -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    db_path = tmp_path / "training.sqlite3"
    output = tmp_path / "training.csv"
    image_path = tmp_path / "data" / "raw" / "mount_tam_east_peak" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(100, 120, 140)).save(image_path)

    db.init_db(db_path)
    with db.connect(db_path) as conn:
        db.register_config(conn, config)
        first = insert_capture_with_image(conn, image_path, "2026-07-08T12:00:00+00:00", "a")
        too_close = insert_capture_with_image(conn, image_path, "2026-07-08T12:01:00+00:00", "b")
        spaced = insert_capture_with_image(conn, image_path, "2026-07-08T12:05:00+00:00", "c")
        for capture_id in (first, too_close, spaced):
            add_two_labels(conn, capture_id, "clouds_below_peak", "clouds_below_peak")

        summary = build_training_set(
            conn,
            config=config,
            options=TrainingSetOptions(
                task_id="marine_layer_detection",
                output_path=output,
                exclude_labels=config.task_excluded_training_labels("marine_layer_detection"),
            ),
        )

    csv_rows = list(csv.DictReader(output.open(encoding="utf-8")))
    capture_ids = {int(row["capture_id"]) for row in csv_rows}
    assert summary["candidate_min_spacing_seconds"] == 300
    assert summary["row_count"] == 2
    assert too_close not in capture_ids
    assert first in capture_ids
    assert spaced in capture_ids


def test_stratified_splits_keep_minority_label_in_each_available_split() -> None:
    rows = []
    for index in range(9):
        rows.append(
            {
                "capture_id": index,
                "camera_id": "camera_a",
                "captured_at_utc": f"2026-07-08T00:{index:02d}:00+00:00",
                "label": "clouds_below_peak",
            }
        )
    for index in range(90):
        rows.append(
            {
                "capture_id": 100 + index,
                "camera_id": "camera_a",
                "captured_at_utc": f"2026-07-09T00:{index:02d}:00+00:00",
                "label": "no_clouds_below_peak",
            }
        )

    assign_stratified_chronological_splits(
        rows,
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
    )

    counts = split_label_counts(rows)
    assert counts["train"]["clouds_below_peak"] == 6
    assert counts["val"]["clouds_below_peak"] == 1
    assert counts["test"]["clouds_below_peak"] == 2


def insert_capture_with_image(
    conn,
    image_path: Path,
    captured_at_utc: str,
    sha: str,
) -> int:
    capture_id = db.insert_capture(
        conn,
        camera_id="mount_tam_east_peak",
        pose_version="initial",
        captured_at_utc=captured_at_utc,
        requested_url="https://example.test/frame.jpg",
        http_status=200,
        content_type="image/jpeg",
        byte_count=100,
        sha256=sha,
        error=None,
    )
    db.insert_image_asset(
        conn,
        capture_id=capture_id,
        path=image_path,
        sha256=sha,
        width=8,
        height=8,
    )
    return capture_id


def add_two_labels(conn, capture_id: int, label_a: str, label_b: str) -> None:
    save_annotation(
        conn,
        capture_id=capture_id,
        task_id="marine_layer_detection",
        label=label_a,
        annotator="jesse",
    )
    save_annotation(
        conn,
        capture_id=capture_id,
        task_id="marine_layer_detection",
        label=label_b,
        annotator="partner",
    )
