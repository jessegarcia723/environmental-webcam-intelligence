import sqlite3
from pathlib import Path

from enviro_webcam_ml.backup import backup_sqlite_database


def test_backup_sqlite_database(tmp_path: Path) -> None:
    source = tmp_path / "source.sqlite3"
    backup = tmp_path / "backup.sqlite3"
    conn = sqlite3.connect(source)
    try:
        conn.execute("CREATE TABLE item (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO item (name) VALUES ('marine layer')")
        conn.commit()
        backup_sqlite_database(source, backup)
    finally:
        conn.close()

    copied = sqlite3.connect(backup)
    try:
        row = copied.execute("SELECT name FROM item").fetchone()
    finally:
        copied.close()

    assert row == ("marine layer",)
