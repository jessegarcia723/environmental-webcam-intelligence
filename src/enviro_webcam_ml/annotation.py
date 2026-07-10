from __future__ import annotations

import json
import mimetypes
import sqlite3
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from enviro_webcam_ml import db
from enviro_webcam_ml.config import AppConfig


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
                image_path = image_path_for_capture(conn, capture_id)
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
    return {
        "capture_id": row["capture_id"],
        "camera_id": row["camera_id"],
        "pose_version": row["pose_version"],
        "captured_at_utc": row["captured_at_utc"],
        "image_url": f"/image/{row['capture_id']}",
        "image_path": row["image_path"],
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


def image_path_for_capture(conn: sqlite3.Connection, capture_id: int) -> str | None:
    row = conn.execute(
        "SELECT path FROM image_asset WHERE capture_id = ? ORDER BY id LIMIT 1",
        (capture_id,),
    ).fetchone()
    return None if row is None else str(row["path"])


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
