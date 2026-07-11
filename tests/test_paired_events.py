from pathlib import Path

from PIL import Image

from enviro_webcam_ml import db
from enviro_webcam_ml.annotation import save_annotation
from enviro_webcam_ml.config import load_config
from enviro_webcam_ml.paired_events import (
    POSITIVE_EVENT_LABEL,
    PairedEventOptions,
    build_paired_events,
)


def test_build_paired_events_writes_summary_visualization_and_gallery(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    db_path = tmp_path / "envirocam.sqlite3"
    data_dir = tmp_path / "data"
    output_dir = data_dir / "reports" / "paired_events"
    write_config(config_path, db_path, data_dir)
    config = load_config(config_path)

    db.init_db(db_path)
    with db.connect(db_path) as conn:
        east_positive = insert_labeled_capture(
            conn,
            data_dir,
            camera_id="east",
            captured_at_utc="2026-07-08T15:00:00+00:00",
            label="clouds_below_peak",
            color=(220, 220, 220),
        )
        west_positive = insert_labeled_capture(
            conn,
            data_dir,
            camera_id="west",
            captured_at_utc="2026-07-08T15:01:00+00:00",
            label="clouds_below_peak",
            color=(210, 210, 210),
        )
        east_negative = insert_labeled_capture(
            conn,
            data_dir,
            camera_id="east",
            captured_at_utc="2026-07-08T16:00:00+00:00",
            label="clouds_below_peak",
            color=(220, 220, 220),
        )
        west_negative = insert_labeled_capture(
            conn,
            data_dir,
            camera_id="west",
            captured_at_utc="2026-07-08T16:01:00+00:00",
            label="no_clouds_below_peak",
            color=(20, 50, 90),
        )
        assert east_positive and west_positive and east_negative and west_negative

        summary = build_paired_events(
            conn,
            config=config,
            options=PairedEventOptions(
                task_id="marine_layer_detection",
                output_dir=output_dir,
                camera_ids=("east", "west"),
                positive_label="clouds_below_peak",
                max_pair_minutes=3,
                thumbnail_width=64,
                timezone_name="America/Los_Angeles",
            ),
        )

    assert summary["event_count"] == 2
    assert summary["both_positive_count"] == 1
    assert summary["event_label_counts"][POSITIVE_EVENT_LABEL] == 1
    assert Path(summary["paths"]["events_csv"]).exists()
    assert Path(summary["paths"]["summary_md"]).exists()
    assert Path(summary["paths"]["hour_csv"]).exists()
    assert Path(summary["paths"]["hour_png"]).exists()
    assert Path(summary["paths"]["examples_html"]).exists()
    assert summary["gallery"]["side_by_side_images_written"] == 1

    events_csv = Path(summary["paths"]["events_csv"]).read_text(encoding="utf-8")
    assert "both_cameras_clouds_below_peak" in events_csv
    assert "not_both_cameras_clouds_below_peak" in events_csv


def insert_labeled_capture(
    conn,
    data_dir: Path,
    *,
    camera_id: str,
    captured_at_utc: str,
    label: str,
    color: tuple[int, int, int],
) -> int:
    image_path = data_dir / "raw" / camera_id / f"{captured_at_utc.replace(':', '')}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (96, 64), color=color).save(image_path)
    capture_id = db.insert_capture(
        conn,
        camera_id=camera_id,
        pose_version="initial",
        captured_at_utc=captured_at_utc,
        requested_url="https://example.test/frame.jpg",
        http_status=200,
        content_type="image/jpeg",
        byte_count=100,
        sha256=f"{camera_id}-{captured_at_utc}",
        error=None,
    )
    db.insert_image_asset(
        conn,
        capture_id=capture_id,
        path=image_path,
        sha256=f"{camera_id}-{captured_at_utc}",
        width=96,
        height=64,
    )
    for annotator in ("Jesse", "Lauren"):
        save_annotation(
            conn,
            capture_id=capture_id,
            task_id="marine_layer_detection",
            label=label,
            annotator=annotator,
        )
    return capture_id


def write_config(config_path: Path, db_path: Path, data_dir: Path) -> None:
    config_path.write_text(
        f"""
project:
  name: test
  database_path: "{db_path}"
  data_dir: "{data_dir}"
cameras:
  - id: east
    name: East
    location:
      latitude: 0
      longitude: 0
      timezone: America/Los_Angeles
    capture:
      image_url: "https://example.test/east.jpg"
    pose:
      version: initial
  - id: west
    name: West
    location:
      latitude: 0
      longitude: 0
      timezone: America/Los_Angeles
    capture:
      image_url: "https://example.test/west.jpg"
    pose:
      version: initial
weather:
  provider: open_meteo
tasks:
  - id: marine_layer_detection
    default: true
    labels:
      - clouds_below_peak
      - no_clouds_below_peak
    positive_label: clouds_below_peak
    comparison_groups:
      camera:
        - east
        - west
""",
        encoding="utf-8",
    )
