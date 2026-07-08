from pathlib import Path

from enviro_webcam_ml.config import load_config


def test_load_mount_tam_config() -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    assert config.project.name == "mount_tam_marine_layer"
    assert config.cameras[0].id == "mount_tam_east_peak"
    assert config.weather.provider == "open_meteo"
