from __future__ import annotations

import json
import mimetypes
import sqlite3
import webbrowser
import csv
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from enviro_webcam_ml import db
from enviro_webcam_ml.config import AppConfig
from enviro_webcam_ml.image_paths import resolve_image_path


DEFAULT_LABELS = [
    "positive",
    "negative",
    "uncertain",
    "bad_frame",
]


@dataclass(frozen=True)
class AnnotationServerOptions:
    host: str
    port: int
    task_id: str
    left_annotator: str
    right_annotator: str
    open_browser: bool = False


@dataclass(frozen=True)
class AdjudicationServerOptions:
    host: str
    port: int
    task_id: str
    adjudicator: str = "adjudicated"
    predictions_csv: Path | None = None
    annotators: tuple[str, ...] = ()
    include_agreements: bool = False
    open_browser: bool = False


def serve_annotation_app(config: AppConfig, options: AnnotationServerOptions) -> None:
    db.init_db(config.database_path)
    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)

    handler_class = make_handler(config, options)
    server = ThreadingHTTPServer((options.host, options.port), handler_class)
    url = f"http://{options.host}:{server.server_port}/"
    print(f"annotation app running at {url}")
    print("Press Control-C to stop.")
    if options.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nannotation app stopped")
    finally:
        server.server_close()


def serve_adjudication_app(config: AppConfig, options: AdjudicationServerOptions) -> None:
    db.init_db(config.database_path)
    with db.connect(config.database_path) as conn:
        db.register_config(conn, config)

    predictions = load_model_predictions(options.predictions_csv) if options.predictions_csv else {}
    handler_class = make_adjudication_handler(config, options, predictions)
    server = ThreadingHTTPServer((options.host, options.port), handler_class)
    url = f"http://{options.host}:{server.server_port}/"
    print(f"adjudication app running at {url}")
    print("Press Control-C to stop.")
    if options.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nadjudication app stopped")
    finally:
        server.server_close()


def make_handler(config: AppConfig, options: AnnotationServerOptions):
    class AnnotationHandler(BaseHTTPRequestHandler):
        server_version = "EnviroCamAnnotation/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature.
            print(f"{self.address_string()} - {format % args}")

        def do_GET(self) -> None:  # noqa: N802 - stdlib method.
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_html(ANNOTATION_HTML)
            elif parsed.path == "/api/config":
                self.send_json(annotation_config_payload(config, options))
            elif parsed.path == "/api/next":
                self.handle_next(parsed.query)
            elif parsed.path == "/api/stats":
                self.handle_stats(parsed.query)
            elif parsed.path.startswith("/image/"):
                self.handle_image(parsed.path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802 - stdlib method.
            parsed = urlparse(self.path)
            if parsed.path == "/api/annotate":
                self.handle_annotate()
            elif parsed.path == "/api/unannotate":
                self.handle_unannotate()
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def handle_next(self, query: str) -> None:
            params = parse_qs(query)
            task_id = first(params, "task_id", options.task_id)
            annotator = first(params, "annotator", "")
            exclude_ids = parse_int_list(first(params, "exclude_ids", ""))
            with db.connect(config.database_path) as conn:
                row = next_unannotated_frame(
                    conn,
                    task_id=task_id,
                    annotator=annotator,
                    exclude_ids=exclude_ids,
                    data_dir=config.data_dir,
                )
            self.send_json({"frame": row})

        def handle_stats(self, query: str) -> None:
            params = parse_qs(query)
            task_id = first(params, "task_id", options.task_id)
            with db.connect(config.database_path) as conn:
                stats = annotation_stats(conn, task_id=task_id)
            self.send_json({"stats": stats})

        def handle_image(self, path: str) -> None:
            capture_id_text = unquote(path.removeprefix("/image/"))
            try:
                capture_id = int(capture_id_text)
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid capture ID")
                return

            with db.connect(config.database_path) as conn:
                image_path = image_path_for_capture(conn, capture_id, data_dir=config.data_dir)
            if image_path is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return

            path_obj = Path(image_path)
            if not path_obj.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Image file missing")
                return

            content_type = mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream"
            payload = path_obj.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def handle_annotate(self) -> None:
            try:
                payload = self.read_json()
                capture_id = int(payload["capture_id"])
                task_id = str(payload.get("task_id") or options.task_id)
                label = str(payload["label"])
                annotator = str(payload.get("annotator") or "")
                confidence = payload.get("confidence")
                notes = payload.get("notes")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid annotation payload: {exc}")
                return

            allowed_labels = task_labels(config, task_id)
            if label not in allowed_labels:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Unknown label: {label}")
                return

            with db.connect(config.database_path) as conn:
                save_annotation(
                    conn,
                    capture_id=capture_id,
                    task_id=task_id,
                    label=label,
                    annotator=annotator,
                    confidence=confidence,
                    notes=notes,
                )
            self.send_json({"ok": True})

        def handle_unannotate(self) -> None:
            try:
                payload = self.read_json()
                capture_id = int(payload["capture_id"])
                task_id = str(payload.get("task_id") or options.task_id)
                annotator = str(payload.get("annotator") or "")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid unannotation payload: {exc}")
                return

            with db.connect(config.database_path) as conn:
                deleted = delete_annotation(
                    conn,
                    capture_id=capture_id,
                    task_id=task_id,
                    annotator=annotator,
                )
            self.send_json({"ok": True, "deleted": deleted})

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def send_html(self, html: str) -> None:
            payload = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return AnnotationHandler


def annotation_config_payload(config: AppConfig, options: AnnotationServerOptions) -> dict[str, Any]:
    return {
        "task_id": options.task_id,
        "labels": task_labels(config, options.task_id),
        "left_annotator": options.left_annotator,
        "right_annotator": options.right_annotator,
        "controls": {
            "keyboard_left": ["1", "2", "3", "4", "5"],
            "keyboard_right": ["6", "7", "8", "9", "0"],
            "xbox": {
                "A": "label 1",
                "B": "label 2",
                "X": "label 3",
                "Y": "label 4",
                "LB": "label 5",
                "RB": "skip",
                "View/Back": "undo last annotation",
            },
        },
    }


def make_adjudication_handler(
    config: AppConfig,
    options: AdjudicationServerOptions,
    predictions: dict[int, dict[str, Any]],
):
    class AdjudicationHandler(BaseHTTPRequestHandler):
        server_version = "EnviroCamAdjudication/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature.
            print(f"{self.address_string()} - {format % args}")

        def do_GET(self) -> None:  # noqa: N802 - stdlib method.
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_html(ADJUDICATION_HTML)
            elif parsed.path == "/api/config":
                self.send_json(adjudication_config_payload(config, options))
            elif parsed.path == "/api/next":
                self.handle_next()
            elif parsed.path == "/api/report":
                self.handle_report()
            elif parsed.path.startswith("/image/"):
                self.handle_image(parsed.path)
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:  # noqa: N802 - stdlib method.
            parsed = urlparse(self.path)
            if parsed.path == "/api/adjudicate":
                self.handle_adjudicate()
            else:
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def handle_next(self) -> None:
            with db.connect(config.database_path) as conn:
                row = next_adjudication_case(
                    conn,
                    task_id=options.task_id,
                    predictions=predictions,
                    annotators=options.annotators,
                    data_dir=config.data_dir,
                    include_agreements=options.include_agreements,
                )
            self.send_json({"case": row})

        def handle_report(self) -> None:
            with db.connect(config.database_path) as conn:
                report = adjudication_report(
                    conn,
                    task_id=options.task_id,
                    annotators=options.annotators,
                )
            self.send_json({"report": report})

        def handle_image(self, path: str) -> None:
            capture_id_text = unquote(path.removeprefix("/image/"))
            try:
                capture_id = int(capture_id_text)
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST, "Invalid capture ID")
                return

            with db.connect(config.database_path) as conn:
                image_path = image_path_for_capture(conn, capture_id, data_dir=config.data_dir)
            if image_path is None:
                self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
                return

            path_obj = Path(image_path)
            if not path_obj.exists():
                self.send_error(HTTPStatus.NOT_FOUND, "Image file missing")
                return

            content_type = mimetypes.guess_type(path_obj.name)[0] or "application/octet-stream"
            payload = path_obj.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def handle_adjudicate(self) -> None:
            try:
                payload = self.read_json()
                capture_id = int(payload["capture_id"])
                label = str(payload["label"])
                notes = payload.get("notes")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Invalid adjudication payload: {exc}")
                return

            allowed_labels = task_labels(config, options.task_id)
            if label not in allowed_labels:
                self.send_error(HTTPStatus.BAD_REQUEST, f"Unknown label: {label}")
                return

            prediction = predictions.get(capture_id, {})
            with db.connect(config.database_path) as conn:
                save_adjudication(
                    conn,
                    capture_id=capture_id,
                    task_id=options.task_id,
                    final_label=label,
                    adjudicator=options.adjudicator,
                    notes=notes,
                    model_label=prediction.get("pred_label"),
                    model_confidence=prediction.get("confidence"),
                )
            self.send_json({"ok": True})

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def send_html(self, html: str) -> None:
            payload = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def send_json(self, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    return AdjudicationHandler


def adjudication_config_payload(config: AppConfig, options: AdjudicationServerOptions) -> dict[str, Any]:
    return {
        "task_id": options.task_id,
        "labels": task_labels(config, options.task_id),
        "adjudicator": options.adjudicator,
        "include_agreements": options.include_agreements,
        "predictions_csv": str(options.predictions_csv) if options.predictions_csv else None,
        "annotators": list(options.annotators),
    }


def task_labels(config: AppConfig, task_id: str) -> list[str]:
    for task in config.raw.get("tasks", []):
        if task.get("id") == task_id:
            labels = task.get("labels")
            if labels:
                return [str(label) for label in labels]

            found: list[str] = []
            if task.get("positive_label"):
                found.append(str(task["positive_label"]))
            found.extend(str(label) for label in task.get("negative_labels", []))
            found.extend(str(label) for label in task.get("uncertain_labels", []))
            return found or DEFAULT_LABELS
    return DEFAULT_LABELS


def next_unannotated_frame(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    annotator: str,
    exclude_ids: list[int] | None = None,
    data_dir: Path | None = None,
) -> dict[str, Any] | None:
    exclude_ids = exclude_ids or []
    exclude_clause = ""
    params: list[Any] = [task_id, annotator]
    if exclude_ids:
        placeholders = ", ".join("?" for _ in exclude_ids)
        exclude_clause = f"AND c.id NOT IN ({placeholders})"
        params.extend(exclude_ids)

    row = conn.execute(
        f"""
        SELECT
          c.id AS capture_id,
          c.camera_id,
          c.pose_version,
          c.captured_at_utc,
          ia.path AS image_path,
          ia.width,
          ia.height,
          fq.avg_luminance,
          fq.blur_variance,
          fq.is_night,
          fq.is_blurry,
          fq.is_duplicate,
          fq.flags_json
        FROM capture c
        JOIN image_asset ia ON ia.capture_id = c.id
        LEFT JOIN frame_quality fq ON fq.capture_id = c.id
        WHERE c.error IS NULL
          AND NOT EXISTS (
            SELECT 1
            FROM annotation a
            WHERE a.capture_id = c.id
              AND a.task_id = ?
              AND COALESCE(a.annotator, '') = ?
          )
          {exclude_clause}
        ORDER BY c.captured_at_utc, c.camera_id
        LIMIT 1
        """,
        params,
    ).fetchone()
    if row is None:
        return None
    flags = row["flags_json"]
    image_path = resolved_or_stored_image_path(row["image_path"], data_dir)
    return {
        "capture_id": row["capture_id"],
        "camera_id": row["camera_id"],
        "pose_version": row["pose_version"],
        "captured_at_utc": row["captured_at_utc"],
        "image_url": f"/image/{row['capture_id']}",
        "image_path": image_path,
        "width": row["width"],
        "height": row["height"],
        "avg_luminance": row["avg_luminance"],
        "blur_variance": row["blur_variance"],
        "is_night": bool(row["is_night"]) if row["is_night"] is not None else None,
        "is_blurry": bool(row["is_blurry"]) if row["is_blurry"] is not None else None,
        "is_duplicate": bool(row["is_duplicate"]) if row["is_duplicate"] is not None else None,
        "quality_flags": json.loads(flags) if flags else {},
    }


def save_annotation(
    conn: sqlite3.Connection,
    *,
    capture_id: int,
    task_id: str,
    label: str,
    annotator: str,
    confidence: float | None = None,
    notes: str | None = None,
) -> None:
    now = db.utc_now_iso()
    cur = conn.execute(
        """
        UPDATE annotation
        SET label = ?, confidence = ?, notes = ?, created_at_utc = ?
        WHERE capture_id = ?
          AND task_id = ?
          AND COALESCE(annotator, '') = ?
        """,
        (label, confidence, notes, now, capture_id, task_id, annotator),
    )
    if cur.rowcount:
        return
    conn.execute(
        """
        INSERT INTO annotation (
          capture_id, task_id, label, annotator, confidence, notes, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (capture_id, task_id, label, annotator, confidence, notes, now),
    )


def delete_annotation(
    conn: sqlite3.Connection,
    *,
    capture_id: int,
    task_id: str,
    annotator: str,
) -> bool:
    cur = conn.execute(
        """
        DELETE FROM annotation
        WHERE capture_id = ?
          AND task_id = ?
          AND COALESCE(annotator, '') = ?
        """,
        (capture_id, task_id, annotator),
    )
    return bool(cur.rowcount)


def annotation_stats(conn: sqlite3.Connection, *, task_id: str) -> dict[str, Any]:
    label_rows = conn.execute(
        """
        SELECT COALESCE(annotator, '') AS annotator, label, COUNT(*) AS count
        FROM annotation
        WHERE task_id = ?
        GROUP BY COALESCE(annotator, ''), label
        ORDER BY annotator, label
        """,
        (task_id,),
    ).fetchall()
    total_rows = conn.execute(
        """
        SELECT COALESCE(annotator, '') AS annotator, COUNT(*) AS count
        FROM annotation
        WHERE task_id = ?
        GROUP BY COALESCE(annotator, '')
        ORDER BY annotator
        """,
        (task_id,),
    ).fetchall()
    return {
        "totals": [dict(row) for row in total_rows],
        "by_label": [dict(row) for row in label_rows],
    }


def next_adjudication_case(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    predictions: dict[int, dict[str, Any]] | None = None,
    annotators: tuple[str, ...] = (),
    data_dir: Path | None = None,
    include_agreements: bool = False,
) -> dict[str, Any] | None:
    predictions = predictions or {}
    for case in adjudication_cases(
        conn,
        task_id=task_id,
        predictions=predictions,
        annotators=annotators,
        data_dir=data_dir,
    ):
        if case["already_adjudicated"]:
            continue
        if not include_agreements and case["agreement"]:
            continue
        return case
    return None


def adjudication_cases(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    predictions: dict[int, dict[str, Any]] | None = None,
    annotators: tuple[str, ...] = (),
    data_dir: Path | None = None,
) -> list[dict[str, Any]]:
    predictions = predictions or {}
    annotator_filter = set(annotators)
    rows = conn.execute(
        """
        SELECT
          a.capture_id,
          COALESCE(a.annotator, '') AS annotator,
          a.label,
          a.created_at_utc AS annotated_at_utc,
          c.camera_id,
          c.pose_version,
          c.captured_at_utc,
          ia.path AS image_path,
          ia.width,
          ia.height,
          adj.final_label,
          adj.adjudicator,
          adj.notes AS adjudication_notes,
          adj.created_at_utc AS adjudicated_at_utc
        FROM annotation a
        JOIN capture c ON c.id = a.capture_id
        JOIN image_asset ia ON ia.capture_id = c.id
        LEFT JOIN annotation_adjudication adj
          ON adj.capture_id = a.capture_id
         AND adj.task_id = a.task_id
        WHERE a.task_id = ?
          AND c.error IS NULL
        ORDER BY c.captured_at_utc, a.capture_id, annotator, a.created_at_utc
        """,
        (task_id,),
    ).fetchall()

    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        capture_id = int(row["capture_id"])
        image_path = resolved_or_stored_image_path(row["image_path"], data_dir)
        case = grouped.setdefault(
            capture_id,
            {
                "capture_id": capture_id,
                "camera_id": row["camera_id"],
                "pose_version": row["pose_version"],
                "captured_at_utc": row["captured_at_utc"],
                "image_url": f"/image/{capture_id}",
                "image_path": image_path,
                "width": row["width"],
                "height": row["height"],
                "annotations": {},
                "final_label": row["final_label"],
                "adjudicator": row["adjudicator"],
                "adjudication_notes": row["adjudication_notes"],
                "adjudicated_at_utc": row["adjudicated_at_utc"],
                "model_prediction": predictions.get(capture_id),
            },
        )
        annotator = row["annotator"] or ""
        if annotator_filter and annotator not in annotator_filter:
            continue
        existing = case["annotations"].get(annotator)
        if existing is None or (row["annotated_at_utc"] or "") >= (existing.get("annotated_at_utc") or ""):
            case["annotations"][annotator] = {
                "label": row["label"],
                "annotated_at_utc": row["annotated_at_utc"],
            }

    cases = []
    for case in grouped.values():
        labels = {item["label"] for item in case["annotations"].values()}
        annotator_count = len(case["annotations"])
        if annotator_count < 2:
            continue
        case["labels"] = sorted(labels)
        case["annotator_count"] = annotator_count
        case["agreement"] = len(labels) == 1
        case["already_adjudicated"] = case["final_label"] is not None
        cases.append(case)
    return sorted(cases, key=lambda item: (item["captured_at_utc"], item["camera_id"], item["capture_id"]))


def save_adjudication(
    conn: sqlite3.Connection,
    *,
    capture_id: int,
    task_id: str,
    final_label: str,
    adjudicator: str,
    notes: str | None = None,
    model_label: str | None = None,
    model_confidence: float | str | None = None,
) -> None:
    confidence = parse_optional_float(model_confidence)
    conn.execute(
        """
        INSERT INTO annotation_adjudication (
          capture_id, task_id, final_label, adjudicator, notes,
          model_label, model_confidence, created_at_utc
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(capture_id, task_id) DO UPDATE SET
          final_label = excluded.final_label,
          adjudicator = excluded.adjudicator,
          notes = excluded.notes,
          model_label = excluded.model_label,
          model_confidence = excluded.model_confidence,
          created_at_utc = excluded.created_at_utc
        """,
        (
            capture_id,
            task_id,
            final_label,
            adjudicator,
            notes,
            model_label,
            confidence,
            db.utc_now_iso(),
        ),
    )


def adjudication_report(
    conn: sqlite3.Connection,
    *,
    task_id: str,
    annotators: tuple[str, ...] = (),
) -> dict[str, Any]:
    cases = adjudication_cases(conn, task_id=task_id, annotators=annotators)
    total = len(cases)
    agreements = sum(1 for case in cases if case["agreement"])
    disagreements = total - agreements
    adjudicated = sum(1 for case in cases if case["already_adjudicated"])
    final_counts: dict[str, int] = {}
    for case in cases:
        if case["final_label"]:
            final_counts[case["final_label"]] = final_counts.get(case["final_label"], 0) + 1
    return {
        "task_id": task_id,
        "double_labeled": total,
        "agreements": agreements,
        "disagreements": disagreements,
        "adjudicated": adjudicated,
        "remaining_disagreements": sum(
            1 for case in cases if not case["agreement"] and not case["already_adjudicated"]
        ),
        "final_label_counts": dict(sorted(final_counts.items())),
    }


def load_model_predictions(path: Path | None) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    predictions: dict[int, dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                capture_id = int(row.get("capture_id") or "")
            except ValueError:
                continue
            predictions[capture_id] = {
                "split": row.get("split", ""),
                "true_label": row.get("true_label", ""),
                "pred_label": row.get("pred_label", ""),
                "confidence": parse_optional_float(row.get("confidence")),
                "correct": row.get("correct", ""),
            }
    return predictions


def parse_optional_float(value: float | str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def image_path_for_capture(
    conn: sqlite3.Connection,
    capture_id: int,
    *,
    data_dir: Path | None = None,
) -> str | None:
    row = conn.execute(
        "SELECT path FROM image_asset WHERE capture_id = ? ORDER BY id LIMIT 1",
        (capture_id,),
    ).fetchone()
    if row is None:
        return None
    return resolved_or_stored_image_path(row["path"], data_dir)


def resolved_or_stored_image_path(stored_path: str | None, data_dir: Path | None) -> str:
    if data_dir is None:
        return str(stored_path or "")
    resolved = resolve_image_path(stored_path, data_dir)
    return str(resolved) if resolved is not None else str(stored_path or "")


def first(params: dict[str, list[str]], key: str, default: str) -> str:
    values = params.get(key)
    if not values:
        return default
    return values[0]


def parse_int_list(raw: str) -> list[int]:
    if not raw:
        return []
    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


ANNOTATION_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EnviroCam Annotation</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0b1020;
      --panel: #121a2e;
      --panel2: #18223a;
      --text: #f3f7ff;
      --muted: #9fb0ca;
      --accent: #76e4a6;
      --warn: #ffd166;
      --bad: #ff6b6b;
      --border: #2b3857;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding: 0.75rem 1rem;
      background: #070b16;
      border-bottom: 1px solid var(--border);
    }
    header h1 {
      font-size: 1rem;
      margin: 0;
      letter-spacing: 0.02em;
    }
    header .status {
      color: var(--muted);
      font-size: 0.85rem;
    }
    main {
      display: grid;
      grid-template-columns: 1fr 1fr;
      height: calc(100vh - 48px);
    }
    .pane {
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr auto;
      border-right: 1px solid var(--border);
      background: var(--panel);
    }
    .pane:last-child { border-right: 0; }
    .pane-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding: 0.75rem 1rem;
      background: var(--panel2);
      border-bottom: 1px solid var(--border);
    }
    .pane-title {
      display: flex;
      flex-direction: column;
      gap: 0.15rem;
    }
    .pane-title strong { font-size: 1rem; }
    .pane-title span { color: var(--muted); font-size: 0.8rem; }
    .gamepad {
      color: var(--warn);
      font-size: 0.8rem;
      white-space: nowrap;
    }
    .gamepad.connected { color: var(--accent); }
    .image-wrap {
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 0.75rem;
      background: #050812;
    }
    img.frame {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #000;
    }
    .empty {
      color: var(--muted);
      text-align: center;
      line-height: 1.5;
      padding: 2rem;
    }
    .controls {
      padding: 0.75rem;
      border-top: 1px solid var(--border);
      display: grid;
      gap: 0.6rem;
    }
    .metadata {
      color: var(--muted);
      font-size: 0.78rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
    }
    .legend {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.4rem;
      color: var(--muted);
      font-size: 0.78rem;
      line-height: 1.25;
    }
    .legend span {
      display: block;
      padding: 0.35rem 0.45rem;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: #10182a;
    }
    .legend kbd {
      color: var(--text);
      background: #263250;
      border: 1px solid #3a4a70;
      border-radius: 4px;
      padding: 0.05rem 0.28rem;
      font-size: 0.72rem;
      white-space: nowrap;
    }
    .buttons {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.5rem;
    }
    button {
      appearance: none;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #202b46;
      color: var(--text);
      font: inherit;
      padding: 0.65rem 0.75rem;
      cursor: pointer;
      min-height: 42px;
    }
    button:hover { border-color: var(--accent); }
    button.skip { color: var(--warn); }
    button.undo { color: #93c5fd; }
    .toast {
      min-height: 1.2rem;
      color: var(--accent);
      font-size: 0.85rem;
    }
    .toast.error { color: var(--bad); }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; height: auto; }
      .pane { min-height: 90vh; border-right: 0; border-bottom: 1px solid var(--border); }
    }
  </style>
</head>
<body>
  <header>
    <h1>EnviroCam Annotation</h1>
    <div class="status" id="globalStatus">Loading…</div>
  </header>
  <main>
    <section class="pane" id="leftPane">
      <div class="pane-head">
        <div class="pane-title"><strong>Left Annotator</strong><span id="leftAnnotator"></span></div>
        <div class="gamepad" id="leftGamepad">Gamepad 1: waiting</div>
      </div>
      <div class="image-wrap" id="leftImageWrap"></div>
      <div class="controls">
        <div class="metadata" id="leftMeta"></div>
        <div class="legend" id="leftLegend"></div>
        <div class="buttons" id="leftButtons"></div>
        <div class="toast" id="leftToast"></div>
      </div>
    </section>

    <section class="pane" id="rightPane">
      <div class="pane-head">
        <div class="pane-title"><strong>Right Annotator</strong><span id="rightAnnotator"></span></div>
        <div class="gamepad" id="rightGamepad">Gamepad 2: waiting</div>
      </div>
      <div class="image-wrap" id="rightImageWrap"></div>
      <div class="controls">
        <div class="metadata" id="rightMeta"></div>
        <div class="legend" id="rightLegend"></div>
        <div class="buttons" id="rightButtons"></div>
        <div class="toast" id="rightToast"></div>
      </div>
    </section>
  </main>

  <script>
    const state = {
      taskId: null,
      labels: [],
      panes: {
        left: { annotator: null, frame: null, history: [], lastButtons: [] },
        right: { annotator: null, frame: null, history: [], lastButtons: [] },
      },
    };

    const keyboard = {
      left: ["1", "2", "3", "4", "5"],
      right: ["6", "7", "8", "9", "0"],
    };

    const xboxButtons = [0, 1, 2, 3, 4]; // A, B, X, Y, LB
    const skipButton = 5; // RB
    const undoButton = 8; // View/Back on standard Xbox mappings
    const xboxNames = ["A", "B", "X", "Y", "LB"];

    async function init() {
      const config = await fetchJson("/api/config");
      state.taskId = config.task_id;
      state.labels = config.labels;
      state.panes.left.annotator = config.left_annotator;
      state.panes.right.annotator = config.right_annotator;
      document.getElementById("leftAnnotator").textContent = config.left_annotator;
      document.getElementById("rightAnnotator").textContent = config.right_annotator;
      document.getElementById("globalStatus").textContent =
        `Task: ${config.task_id} · Press any button on each Xbox controller to wake browser detection.`;
      renderLegend("left");
      renderLegend("right");
      renderButtons("left");
      renderButtons("right");
      await Promise.all([loadNext("left"), loadNext("right")]);
      window.addEventListener("keydown", onKeydown);
      window.addEventListener("gamepadconnected", updateGamepadLabels);
      window.addEventListener("gamepaddisconnected", updateGamepadLabels);
      requestAnimationFrame(pollGamepads);
      setInterval(refreshStats, 10000);
      refreshStats();
    }

    function renderButtons(side) {
      const target = document.getElementById(`${side}Buttons`);
      target.innerHTML = "";
      state.labels.forEach((label, index) => {
        const button = document.createElement("button");
        const key = keyboard[side][index] || "";
        const xbox = xboxNames[index] || "";
        button.textContent = `${key ? key + " · " : ""}${xbox ? xbox + " · " : ""}${pretty(label)}`;
        button.addEventListener("click", () => annotate(side, label));
        target.appendChild(button);
      });
      const skip = document.createElement("button");
      skip.className = "skip";
      skip.textContent = side === "left" ? "Q · RB · Skip" : "P · RB · Skip";
      skip.addEventListener("click", () => skipFrame(side));
      target.appendChild(skip);
      const undo = document.createElement("button");
      undo.className = "undo";
      undo.textContent = side === "left" ? "Z · View · Undo" : "O · View · Undo";
      undo.addEventListener("click", () => undoLast(side));
      target.appendChild(undo);
    }

    function renderLegend(side) {
      const target = document.getElementById(`${side}Legend`);
      const keyList = keyboard[side];
      const labelHints = state.labels.map((label, index) => {
        const key = keyList[index] || "—";
        const xbox = xboxNames[index] || "—";
        return `<span><kbd>${escapeHtml(key)}</kbd> / <kbd>${escapeHtml(xbox)}</kbd> ${escapeHtml(pretty(label))}</span>`;
      });
      const skipKey = side === "left" ? "Q" : "P";
      const undoKey = side === "left" ? "Z" : "O";
      target.innerHTML = [
        ...labelHints,
        `<span><kbd>${skipKey}</kbd> / <kbd>RB</kbd> skip frame</span>`,
        `<span><kbd>${undoKey}</kbd> / <kbd>View</kbd> undo last saved label</span>`,
      ].join("");
    }

    async function loadNext(side) {
      const pane = state.panes[side];
      const otherSide = side === "left" ? "right" : "left";
      const exclude = [];
      const otherFrame = state.panes[otherSide].frame;
      if (otherFrame) exclude.push(otherFrame.capture_id);
      const params = new URLSearchParams({
        task_id: state.taskId,
        annotator: pane.annotator,
        exclude_ids: exclude.join(","),
      });
      const payload = await fetchJson(`/api/next?${params}`);
      pane.frame = payload.frame;
      renderFrame(side);
    }

    function renderFrame(side) {
      const pane = state.panes[side];
      const wrap = document.getElementById(`${side}ImageWrap`);
      const meta = document.getElementById(`${side}Meta`);
      if (!pane.frame) {
        wrap.innerHTML = `<div class="empty">No unlabeled frames left for ${escapeHtml(pane.annotator)}.<br>Collect more images or switch annotator.</div>`;
        meta.textContent = "";
        return;
      }
      const frame = pane.frame;
      wrap.innerHTML = `<img class="frame" src="${frame.image_url}" alt="capture ${frame.capture_id}">`;
      const flags = Object.keys(frame.quality_flags || {});
      meta.innerHTML = [
        `#${frame.capture_id}`,
        escapeHtml(frame.camera_id),
        escapeHtml(frame.captured_at_utc),
        frame.is_night ? "night" : null,
        frame.is_blurry ? "blurry" : null,
        frame.is_duplicate ? "duplicate" : null,
        flags.length ? `flags: ${flags.map(escapeHtml).join(", ")}` : null,
      ].filter(Boolean).map(item => `<span>${item}</span>`).join("");
    }

    async function annotate(side, label) {
      const pane = state.panes[side];
      if (!pane.frame) return;
      const captureId = pane.frame.capture_id;
      try {
        await postJson("/api/annotate", {
          capture_id: captureId,
          task_id: state.taskId,
          label,
          annotator: pane.annotator,
        });
        pane.history.push({ captureId, label });
        toast(side, `Saved ${pretty(label)} for #${captureId}`);
        await loadNext(side);
        refreshStats();
      } catch (error) {
        toast(side, error.message, true);
      }
    }

    async function skipFrame(side) {
      toast(side, "Skipped");
      await loadNext(side);
    }

    async function undoLast(side) {
      const pane = state.panes[side];
      const last = pane.history.pop();
      if (!last) {
        toast(side, "Nothing to undo yet", true);
        return;
      }
      try {
        await postJson("/api/unannotate", {
          capture_id: last.captureId,
          task_id: state.taskId,
          annotator: pane.annotator,
        });
        toast(side, `Undid ${pretty(last.label)} for #${last.captureId}`);
        await loadNext(side);
        refreshStats();
      } catch (error) {
        pane.history.push(last);
        toast(side, error.message, true);
      }
    }

    function onKeydown(event) {
      const key = event.key;
      if (key.toLowerCase() === "q") {
        skipFrame("left");
        return;
      }
      if (key.toLowerCase() === "p") {
        skipFrame("right");
        return;
      }
      if (key.toLowerCase() === "z") {
        undoLast("left");
        return;
      }
      if (key.toLowerCase() === "o") {
        undoLast("right");
        return;
      }
      for (const side of ["left", "right"]) {
        const index = keyboard[side].indexOf(key);
        if (index >= 0 && index < state.labels.length) {
          annotate(side, state.labels[index]);
          return;
        }
      }
    }

    function pollGamepads() {
      const pads = connectedGamepads();
      updateGamepadLabels();
      handleGamepad("left", pads[0]);
      handleGamepad("right", pads[1]);
      requestAnimationFrame(pollGamepads);
    }

    function handleGamepad(side, pad) {
      if (!pad) return;
      const pane = state.panes[side];
      const pressed = pad.buttons.map(button => button.pressed);
      xboxButtons.forEach((buttonIndex, labelIndex) => {
        if (pressed[buttonIndex] && !pane.lastButtons[buttonIndex] && labelIndex < state.labels.length) {
          annotate(side, state.labels[labelIndex]);
        }
      });
      if (pressed[skipButton] && !pane.lastButtons[skipButton]) {
        skipFrame(side);
      }
      if (pressed[undoButton] && !pane.lastButtons[undoButton]) {
        undoLast(side);
      }
      pane.lastButtons = pressed;
    }

    function updateGamepadLabels() {
      const pads = connectedGamepads();
      setGamepadText("left", pads[0]);
      setGamepadText("right", pads[1]);
    }

    function connectedGamepads() {
      if (!navigator.getGamepads) return [];
      return Array.from(navigator.getGamepads()).filter(Boolean);
    }

    function setGamepadText(side, pad) {
      const el = document.getElementById(`${side}Gamepad`);
      const number = side === "left" ? 1 : 2;
      if (pad) {
        el.textContent = `Gamepad ${number}: ${pad.id} · browser index ${pad.index}`;
        el.classList.add("connected");
      } else {
        el.textContent = `Gamepad ${number}: waiting — press A/B/X/Y`;
        el.classList.remove("connected");
      }
    }

    async function refreshStats() {
      try {
        const params = new URLSearchParams({ task_id: state.taskId });
        const payload = await fetchJson(`/api/stats?${params}`);
        const totals = (payload.stats.totals || []).map(row => `${row.annotator || "unknown"}: ${row.count}`).join(" · ");
        if (totals) {
          const count = connectedGamepads().length;
          document.getElementById("globalStatus").textContent =
            `Task: ${state.taskId} · ${totals} · gamepads detected: ${count}`;
        }
      } catch (_) {
        // Stats are helpful, not critical.
      }
    }

    function toast(side, message, isError = false) {
      const el = document.getElementById(`${side}Toast`);
      el.textContent = message;
      el.classList.toggle("error", isError);
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function pretty(label) {
      return label.replaceAll("_", " ");
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    init().catch(error => {
      document.getElementById("globalStatus").textContent = error.message;
    });
  </script>
</body>
</html>
"""


ADJUDICATION_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EnviroCam Adjudication</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08111f;
      --panel: #111b2f;
      --panel2: #18243b;
      --text: #f4f8ff;
      --muted: #9fb0ca;
      --accent: #7dd3fc;
      --good: #86efac;
      --warn: #fde68a;
      --bad: #fca5a5;
      --border: #2d3c5f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      padding: 0.75rem 1rem;
      background: #050a14;
      border-bottom: 1px solid var(--border);
    }
    h1 { font-size: 1rem; margin: 0; }
    .status { color: var(--muted); font-size: 0.85rem; }
    main {
      display: grid;
      grid-template-columns: minmax(0, 1.8fr) minmax(360px, 0.8fr);
      height: calc(100vh - 48px);
    }
    .image-wrap {
      min-height: 0;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1rem;
      background: #020713;
    }
    img.frame {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #000;
    }
    aside {
      min-width: 0;
      overflow: auto;
      padding: 1rem;
      background: var(--panel);
      border-left: 1px solid var(--border);
      display: grid;
      align-content: start;
      gap: 1rem;
    }
    .card {
      background: var(--panel2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.9rem;
    }
    .card h2 {
      margin: 0 0 0.6rem;
      font-size: 0.95rem;
    }
    .meta, .report, .annotations {
      color: var(--muted);
      font-size: 0.86rem;
      line-height: 1.45;
    }
    .annotations {
      display: grid;
      gap: 0.45rem;
    }
    .pill {
      display: inline-flex;
      gap: 0.35rem;
      align-items: center;
      padding: 0.24rem 0.5rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #0f172a;
      color: var(--text);
      margin: 0.1rem;
    }
    .pill.good { border-color: #3f7d55; color: var(--good); }
    .pill.bad { border-color: #7f4545; color: var(--bad); }
    .pill.model { border-color: #38658a; color: var(--accent); }
    .buttons {
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.45rem;
    }
    button {
      appearance: none;
      border: 1px solid var(--border);
      border-radius: 9px;
      background: #1f2c49;
      color: var(--text);
      font: inherit;
      padding: 0.65rem 0.75rem;
      cursor: pointer;
      text-align: left;
    }
    button:hover { border-color: var(--accent); }
    button.primary { background: #143356; border-color: #2e7bb5; }
    textarea {
      width: 100%;
      min-height: 5rem;
      resize: vertical;
      border-radius: 9px;
      border: 1px solid var(--border);
      background: #081225;
      color: var(--text);
      padding: 0.7rem;
      font: inherit;
    }
    .empty {
      color: var(--muted);
      text-align: center;
      line-height: 1.5;
      padding: 2rem;
    }
    .toast { color: var(--good); min-height: 1.2rem; font-size: 0.86rem; }
    .toast.error { color: var(--bad); }
    kbd {
      background: #263653;
      border: 1px solid #3a4a70;
      border-radius: 4px;
      padding: 0.05rem 0.3rem;
      font-size: 0.75rem;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; height: auto; }
      .image-wrap { min-height: 55vh; }
      aside { border-left: 0; border-top: 1px solid var(--border); }
    }
  </style>
</head>
<body>
  <header>
    <h1>EnviroCam Adjudication</h1>
    <div class="status" id="status">Loading…</div>
  </header>
  <main>
    <div class="image-wrap" id="imageWrap"></div>
    <aside>
      <section class="card">
        <h2>Current case</h2>
        <div class="meta" id="meta"></div>
      </section>
      <section class="card">
        <h2>Human labels</h2>
        <div class="annotations" id="annotations"></div>
      </section>
      <section class="card">
        <h2>ML prediction</h2>
        <div class="meta" id="prediction"></div>
      </section>
      <section class="card">
        <h2>Final label</h2>
        <div class="buttons" id="buttons"></div>
      </section>
      <section class="card">
        <h2>Notes</h2>
        <textarea id="notes" placeholder="Optional: why did you choose the final label?"></textarea>
      </section>
      <section class="card">
        <h2>Report</h2>
        <div class="report" id="report"></div>
      </section>
      <div class="toast" id="toast"></div>
    </aside>
  </main>

  <script>
    const state = { taskId: null, labels: [], current: null };
    const keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"];

    async function init() {
      const config = await fetchJson("/api/config");
      state.taskId = config.task_id;
      state.labels = config.labels;
      const annotators = (config.annotators || []).length ? ` · annotators: ${config.annotators.join(", ")}` : "";
      document.getElementById("status").textContent =
        `Task: ${config.task_id} · adjudicator: ${config.adjudicator}${annotators} · mode: ${config.include_agreements ? "all double-labeled" : "disagreements only"}`;
      renderButtons();
      window.addEventListener("keydown", onKeydown);
      await refreshReport();
      await loadNext();
    }

    function renderButtons() {
      const target = document.getElementById("buttons");
      target.innerHTML = "";
      state.labels.forEach((label, index) => {
        const button = document.createElement("button");
        const key = keys[index] || "";
        button.className = "primary";
        button.textContent = `${key ? key + " · " : ""}${pretty(label)}`;
        button.addEventListener("click", () => saveFinal(label));
        target.appendChild(button);
      });
    }

    async function loadNext() {
      const payload = await fetchJson("/api/next");
      state.current = payload.case;
      renderCase();
    }

    function renderCase() {
      const imageWrap = document.getElementById("imageWrap");
      const meta = document.getElementById("meta");
      const annotations = document.getElementById("annotations");
      const prediction = document.getElementById("prediction");
      document.getElementById("notes").value = "";
      if (!state.current) {
        imageWrap.innerHTML = '<div class="empty">No adjudication cases left. Tiny confetti in the database. 🎉</div>';
        meta.textContent = "";
        annotations.textContent = "";
        prediction.textContent = "";
        return;
      }
      const item = state.current;
      imageWrap.innerHTML = `<img class="frame" src="${item.image_url}" alt="capture ${item.capture_id}">`;
      meta.innerHTML = [
        `#${item.capture_id}`,
        escapeHtml(item.camera_id),
        escapeHtml(item.captured_at_utc),
        item.agreement ? '<span class="pill good">agreement</span>' : '<span class="pill bad">disagreement</span>',
      ].filter(Boolean).map(value => `<div>${value}</div>`).join("");
      annotations.innerHTML = Object.entries(item.annotations || {})
        .map(([annotator, row]) => `<div><span class="pill">${escapeHtml(annotator || "unknown")}</span> ${escapeHtml(pretty(row.label))}</div>`)
        .join("");
      const model = item.model_prediction;
      if (model && model.pred_label) {
        const confidence = model.confidence == null ? "n/a" : Number(model.confidence).toFixed(4);
        prediction.innerHTML = [
          `<span class="pill model">${escapeHtml(pretty(model.pred_label))}</span>`,
          `confidence: ${escapeHtml(confidence)}`,
          model.split ? `split: ${escapeHtml(model.split)}` : null,
        ].filter(Boolean).map(value => `<div>${value}</div>`).join("");
      } else {
        prediction.innerHTML = '<span class="pill">No model prediction loaded for this capture</span>';
      }
    }

    async function saveFinal(label) {
      if (!state.current) return;
      try {
        await postJson("/api/adjudicate", {
          capture_id: state.current.capture_id,
          label,
          notes: document.getElementById("notes").value,
        });
        toast(`Saved final label ${pretty(label)} for #${state.current.capture_id}`);
        await refreshReport();
        await loadNext();
      } catch (error) {
        toast(error.message, true);
      }
    }

    async function refreshReport() {
      const payload = await fetchJson("/api/report");
      const report = payload.report;
      document.getElementById("report").innerHTML = [
        `double-labeled: ${report.double_labeled}`,
        `agreements: ${report.agreements}`,
        `disagreements: ${report.disagreements}`,
        `adjudicated: ${report.adjudicated}`,
        `remaining disagreements: ${report.remaining_disagreements}`,
        `final labels: ${escapeHtml(JSON.stringify(report.final_label_counts))}`,
      ].map(value => `<div>${value}</div>`).join("");
    }

    function onKeydown(event) {
      const index = keys.indexOf(event.key);
      if (index >= 0 && index < state.labels.length) {
        saveFinal(state.labels[index]);
      }
    }

    async function fetchJson(url) {
      const response = await fetch(url);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    async function postJson(url, payload) {
      const response = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function toast(message, isError = false) {
      const el = document.getElementById("toast");
      el.textContent = message;
      el.classList.toggle("error", isError);
    }

    function pretty(label) {
      return String(label || "").replaceAll("_", " ");
    }

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    init().catch(error => {
      document.getElementById("status").textContent = error.message;
    });
  </script>
</body>
</html>
"""
