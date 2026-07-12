from pathlib import Path

from enviro_webcam_ml.config import load_config


def test_load_mount_tam_config() -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    assert config.project.name == "mount_tam_marine_layer"
    assert config.cameras[0].id == "mount_tam_east_peak"
    assert config.weather.provider == "open_meteo"
    assert config.weather.fetch_interval_seconds == 3600
    assert "boundary_layer_height" in config.weather.hourly_variables
    assert config.default_task_id == "marine_layer_detection"
    assert config.cameras[0].capture.source == "manifest_frames"
    assert config.cameras[0].capture.interval_seconds == 3600
    assert config.cameras[0].capture.manifest is not None
    assert config.cameras[0].capture.manifest.min_frame_spacing_seconds == 300
    assert config.cameras[0].capture.manifest.frames_key == "frames"
    assert config.task_excluded_training_labels() == ("night_unusable", "camera_artifact")
    assert config.task_comparison_camera_ids() == ("mount_tam_east_peak", "mount_tam_west_peak")
    assert config.task_training_csv_path().name == "marine_layer_detection_training.csv"
    assert config.task_positive_label() == "clouds_below_peak"
    assert config.task_positive_threshold() is None
    assert config.task_class_weights() == {}
    assert config.task_image_crop_pixels().as_dict() == {
        "top": 160,
        "bottom": 31,
        "left": 0,
        "right": 0,
    }


def test_training_config_expands_data_dir_env_var(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ENVIROCAM_DATA_DIR", str(tmp_path))

    config = load_config(Path("configs/mount_tam_training.yaml"))

    assert config.data_dir == tmp_path
    assert config.database_path == tmp_path / "mount_tam.sqlite3"
