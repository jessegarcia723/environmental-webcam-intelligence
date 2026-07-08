from pathlib import Path

from enviro_webcam_ml import db
from enviro_webcam_ml.config import load_config
from enviro_webcam_ml.dataset import build_manifest


def test_init_db_and_empty_manifest(tmp_path: Path) -> None:
    config = load_config(Path("configs/mount_tam.yaml"))
    db_path = tmp_path / "test.sqlite3"
    db.init_db(db_path)
    with db.connect(db_path) as conn:
        db.register_config(conn, config)
        output = tmp_path / "manifest.csv"
        count = build_manifest(conn, output)

    assert count == 0
    assert output.read_text(encoding="utf-8").startswith("capture_id,camera_id")
