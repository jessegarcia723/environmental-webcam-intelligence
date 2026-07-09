from __future__ import annotations

import sqlite3
from pathlib import Path


def backup_sqlite_database(source_path: Path, output_path: Path) -> None:
    """Create a consistent SQLite backup, even if the source DB is live."""
    if not source_path.exists():
        raise FileNotFoundError(f"Database does not exist: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(source_path)
    try:
        destination = sqlite3.connect(output_path)
        try:
            source.backup(destination)
        finally:
            destination.close()
    finally:
        source.close()
