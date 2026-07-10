from pathlib import Path

from enviro_webcam_ml.config import load_config


def test_load_mount_tam_config() -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    assert config.project.name == "mount_tam_marine_layer"
    assert config.cameras[0].id == "mount_tam_east_peak"
    assert config.weather.provider == "open_meteo"


def test_training_config_expands_data_dir_env_var(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ENVIROCAM_DATA_DIR", str(tmp_path))

    config = load_config(Path("configs/mount_tam_training.yaml"))

    assert config.data_dir == tmp_path
    assert config.database_path == tmp_path / "mount_tam.sqlite3"
