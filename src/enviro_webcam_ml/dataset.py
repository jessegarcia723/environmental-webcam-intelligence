from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path


MANIFEST_QUERY = """
SELECT
  c.id AS capture_id,
  c.camera_id,
  c.pose_version,
  c.captured_at_utc,
  ia.path AS image_path,
  ia.sha256,
  ia.width,
  ia.height,
  fq.avg_luminance,
  fq.blur_variance,
  fq.is_night,
  fq.is_blurry,
  fq.is_duplicate,
  fq.flags_json,
  a.task_id,
  a.label
FROM capture c
LEFT JOIN image_asset ia ON ia.capture_id = c.id
LEFT JOIN frame_quality fq ON fq.capture_id = c.id
LEFT JOIN annotation a ON a.capture_id = c.id
WHERE c.error IS NULL
ORDER BY c.camera_id, c.captured_at_utc
"""


def build_manifest(conn: sqlite3.Connection, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(MANIFEST_QUERY).fetchall()
    fieldnames = [
        "capture_id",
        "camera_id",
        "pose_version",
        "captured_at_utc",
        "image_path",
        "sha256",
        "width",
        "height",
        "avg_luminance",
        "blur_variance",
        "is_night",
        "is_blurry",
        "is_duplicate",
        "quality_flags",
        "task_id",
        "label",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flags = row["flags_json"]
            if flags:
                flags = json.dumps(json.loads(flags), sort_keys=True)
            writer.writerow(
                {
                    "capture_id": row["capture_id"],
                    "camera_id": row["camera_id"],
                    "pose_version": row["pose_version"],
                    "captured_at_utc": row["captured_at_utc"],
                    "image_path": row["image_path"],
                    "sha256": row["sha256"],
                    "width": row["width"],
                    "height": row["height"],
                    "avg_luminance": row["avg_luminance"],
                    "blur_variance": row["blur_variance"],
                    "is_night": row["is_night"],
                    "is_blurry": row["is_blurry"],
                    "is_duplicate": row["is_duplicate"],
                    "quality_flags": flags,
                    "task_id": row["task_id"],
                    "label": row["label"],
                }
            )
    return len(rows)
