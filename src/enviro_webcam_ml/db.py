from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from enviro_webcam_ml.config import AppConfig, CameraConfig


SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS camera (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  latitude REAL NOT NULL,
  longitude REAL NOT NULL,
  elevation_m REAL,
  timezone TEXT NOT NULL,
  image_url TEXT NOT NULL,
  created_at_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS camera_pose (
  camera_id TEXT NOT NULL,
  pose_version TEXT NOT NULL,
  description TEXT,
  created_at_utc TEXT NOT NULL,
  PRIMARY KEY (camera_id, pose_version),
  FOREIGN KEY (camera_id) REFERENCES camera(id)
);

CREATE TABLE IF NOT EXISTS capture (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  camera_id TEXT NOT NULL,
  pose_version TEXT NOT NULL,
  captured_at_utc TEXT NOT NULL,
  requested_url TEXT NOT NULL,
  http_status INTEGER,
  content_type TEXT,
  byte_count INTEGER,
  sha256 TEXT,
  error TEXT,
  created_at_utc TEXT NOT NULL,
  FOREIGN KEY (camera_id) REFERENCES camera(id)
);

CREATE INDEX IF NOT EXISTS idx_capture_camera_time
ON capture(camera_id, captured_at_utc);

CREATE TABLE IF NOT EXISTS image_asset (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  capture_id INTEGER NOT NULL,
  path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  width INTEGER,
  height INTEGER,
  created_at_utc TEXT NOT NULL,
  FOREIGN KEY (capture_id) REFERENCES capture(id)
);

CREATE TABLE IF NOT EXISTS frame_quality (
  capture_id INTEGER PRIMARY KEY,
  avg_luminance REAL,
  blur_variance REAL,
  is_night INTEGER NOT NULL,
  is_blurry INTEGER NOT NULL,
  is_duplicate INTEGER NOT NULL,
  flags_json TEXT NOT NULL,
  created_at_utc TEXT NOT NULL,
  FOREIGN KEY (capture_id) REFERENCES capture(id)
);

CREATE TABLE IF NOT EXISTS weather_raw (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  camera_id TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  url TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  FOREIGN KEY (camera_id) REFERENCES camera(id)
);

CREATE TABLE IF NOT EXISTS weather_record (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  camera_id TEXT NOT NULL,
  valid_at_utc TEXT NOT NULL,
  fetched_at_utc TEXT NOT NULL,
  variables_json TEXT NOT NULL,
  FOREIGN KEY (camera_id) REFERENCES camera(id)
);

CREATE INDEX IF NOT EXISTS idx_weather_camera_valid
ON weather_record(camera_id, valid_at_utc);

CREATE TABLE IF NOT EXISTS annotation (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  capture_id INTEGER NOT NULL,
  task_id TEXT NOT NULL,
  label TEXT NOT NULL,
  annotator TEXT,
  confidence REAL,
  notes TEXT,
  created_at_utc TEXT NOT NULL,
  FOREIGN KEY (capture_id) REFERENCES capture(id)
);

CREATE TABLE IF NOT EXISTS annotation_adjudication (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  capture_id INTEGER NOT NULL,
  task_id TEXT NOT NULL,
  final_label TEXT NOT NULL,
  adjudicator TEXT,
  notes TEXT,
  model_label TEXT,
  model_confidence REAL,
  created_at_utc TEXT NOT NULL,
  UNIQUE(capture_id, task_id),
  FOREIGN KEY (capture_id) REFERENCES capture(id)
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def connect(db_path: str | Path) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def register_config(conn: sqlite3.Connection, config: AppConfig) -> None:
    for camera in config.cameras:
        upsert_camera(conn, camera)


def upsert_camera(conn: sqlite3.Connection, camera: CameraConfig) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        INSERT INTO camera (
          id, name, latitude, longitude, elevation_m, timezone, image_url, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          name = excluded.name,
          latitude = excluded.latitude,
          longitude = excluded.longitude,
          elevation_m = excluded.elevation_m,
          timezone = excluded.timezone,
          image_url = excluded.image_url
        """,
        (
            camera.id,
            camera.name,
            camera.location.latitude,
            camera.location.longitude,
            camera.location.elevation_m,
            camera.location.timezone,
            camera.capture.image_url,
            now,
        ),
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO camera_pose (
          camera_id, pose_version, description, created_at_utc
        )
        VALUES (?, ?, ?, ?)
        """,
        (camera.id, camera.pose.version, camera.pose.description, now),
    )


def insert_capture(
    conn: sqlite3.Connection,
    *,
    camera_id: str,
    pose_version: str,
    captured_at_utc: str,
    requested_url: str,
    http_status: int | None,
    content_type: str | None,
    byte_count: int | None,
    sha256: str | None,
    error: str | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO capture (
          camera_id, pose_version, captured_at_utc, requested_url, http_status,
          content_type, byte_count, sha256, error, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            pose_version,
            captured_at_utc,
            requested_url,
            http_status,
            content_type,
            byte_count,
            sha256,
            error,
            utc_now_iso(),
        ),
    )
    return int(cur.lastrowid)


def insert_image_asset(
    conn: sqlite3.Connection,
    *,
    capture_id: int,
    path: Path,
    sha256: str,
    width: int | None,
    height: int | None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO image_asset (
          capture_id, path, sha256, width, height, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (capture_id, str(path), sha256, width, height, utc_now_iso()),
    )
    return int(cur.lastrowid)


def insert_frame_quality(
    conn: sqlite3.Connection,
    *,
    capture_id: int,
    avg_luminance: float | None,
    blur_variance: float | None,
    is_night: bool,
    is_blurry: bool,
    is_duplicate: bool,
    flags: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO frame_quality (
          capture_id, avg_luminance, blur_variance, is_night, is_blurry,
          is_duplicate, flags_json, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            capture_id,
            avg_luminance,
            blur_variance,
            int(is_night),
            int(is_blurry),
            int(is_duplicate),
            json.dumps(flags, sort_keys=True),
            utc_now_iso(),
        ),
    )


def latest_successful_sha(conn: sqlite3.Connection, camera_id: str, before_capture_id: int) -> str | None:
    row = conn.execute(
        """
        SELECT sha256
        FROM capture
        WHERE camera_id = ?
          AND id < ?
          AND sha256 IS NOT NULL
          AND error IS NULL
        ORDER BY captured_at_utc DESC
        LIMIT 1
        """,
        (camera_id, before_capture_id),
    ).fetchone()
    return None if row is None else str(row["sha256"])


def insert_weather_payload(
    conn: sqlite3.Connection,
    *,
    provider: str,
    camera_id: str,
    fetched_at_utc: str,
    url: str,
    payload: dict[str, Any],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO weather_raw (
          provider, camera_id, fetched_at_utc, url, payload_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (provider, camera_id, fetched_at_utc, url, json.dumps(payload, sort_keys=True)),
    )
    return int(cur.lastrowid)


def insert_weather_records(
    conn: sqlite3.Connection,
    *,
    provider: str,
    camera_id: str,
    fetched_at_utc: str,
    records: list[dict[str, Any]],
) -> int:
    rows = [
        (
            provider,
            camera_id,
            record["valid_at_utc"],
            fetched_at_utc,
            json.dumps(record["variables"], sort_keys=True),
        )
        for record in records
    ]
    conn.executemany(
        """
        INSERT INTO weather_record (
          provider, camera_id, valid_at_utc, fetched_at_utc, variables_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value
