# Environmental Webcam ML MVP

This repository turns the framework in [ENVIRONMENTAL_WEBCAM_ML_FRAMEWORK.md](/Users/jessegarica/Documents/Mount%20Tam/ENVIRONMENTAL_WEBCAM_ML_FRAMEWORK.md) into a runnable first build.

The first milestone is intentionally practical:

- capture frames from one or more fixed webcams;
- store immutable image files plus SQLite metadata;
- fetch normalized weather data from Open-Meteo;
- compute basic image quality signals;
- build manifest CSVs for annotation/training;
- keep detection and prediction data contracts separate so we do not leak future information.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

envirocam init-db --config configs/mount_tam.yaml
envirocam capture-once --config configs/mount_tam.yaml
envirocam capture-loop --config configs/mount_tam.yaml --max-iterations 3
envirocam fetch-weather --config configs/mount_tam.yaml
envirocam run-collector --config configs/mount_tam.yaml --max-iterations 1
envirocam build-manifest --config configs/mount_tam.yaml --output data/manifests/mount_tam_frames.csv
envirocam annotate --config configs/mount_tam.yaml --open-browser
envirocam analyze-annotations --config configs/mount_tam.yaml
pytest
```

The sample config is wired to the Mount Tam still-frame endpoints embedded on the Marin Sonoma Rentals webcam page:

- `Axis-TamEast`: `https://cameras.alertcalifornia.org/public-camera-data/Axis-TamEast/latest-frame.jpg`
- `Axis-TamWest`: `https://cameras.alertcalifornia.org/public-camera-data/Axis-TamWest/latest-frame.jpg`

For a long-running local collector, use `run-collector`. It captures webcam frames every 5 minutes and fetches Open-Meteo weather every 2 hours:

```bash
envirocam run-collector --config configs/mount_tam.yaml
```

On a MacBook, wrap it with `caffeinate` so the machine stays awake:

```bash
caffeinate -dimsu envirocam run-collector --config configs/mount_tam.yaml
```

`run-collector` includes a clock sanity check. It compares system UTC time against Python's monotonic timer before writing data. If your Mac clock freezes, jumps backward, or jumps far ahead, the collector prints a warning, skips that capture/weather cycle, waits 60 seconds, and tries again. This prevents obviously bad timestamps from polluting the dataset.

If you only want webcam images and no scheduled weather fetches, use `capture-loop`:

```bash
envirocam capture-loop --config configs/mount_tam.yaml
```

## Annotation

Run the local split-screen annotation app:

```bash
envirocam annotate \
  --config configs/mount_tam.yaml \
  --left-annotator jesse \
  --right-annotator partner \
  --open-browser
```

Open Chrome to `http://127.0.0.1:8000` if the browser does not open automatically.

Controls:

- Left pane keyboard: `1`, `2`, `3`, `4`, `5`
- Right pane keyboard: `6`, `7`, `8`, `9`, `0`
- Xbox controller per pane: `A`, `B`, `X`, `Y`, `LB`
- Skip current frame: left `Q`, right `P`, or Xbox `RB`
- Undo last saved annotation from the current session: left `Z`, right `O`, or Xbox `View/Back`

The app shows the full keyboard/Xbox mapping inside each annotation pane. Undo is per annotator: if the left annotator undoes a label, it removes only that annotator's row for that image and leaves the other person's annotation untouched.

For two Bluetooth Xbox controllers on a MacBook, pair both controllers in macOS Bluetooth settings before opening the app. Chrome is the recommended browser because its Gamepad API support is the most reliable on macOS. The first connected controller controls the left pane; the second connected controller controls the right pane.

If only one controller appears, press a button on both controllers while the annotation page is focused. Some browsers do not expose a controller to the Gamepad API until it has been touched. The app compacts the browser's connected-controller list, so controllers can appear at raw browser indices like `0` and `2` and still map correctly to left/right panes.

Annotations are saved into `data/mount_tam.sqlite3` in the `annotation` table.

Current Mount Tam labels:

- `clouds_below_peak`
- `no_clouds_below_peak`
- `peak_obscured`
- `below_peak_height_far_from_peak`
- `night_unusable`
- `camera_artifact`

The older labels `peak_obscured_uncertain` and `uncertain` are intentionally no longer part of the active label set. If either appears in analysis reports, re-review those frames rather than automatically mapping them.

## Annotation analysis

Generate a multi-rater annotation report:

```bash
envirocam analyze-annotations --config configs/mount_tam.yaml
```

This writes:

- `data/reports/annotation_analysis.md`
- `data/reports/disagreements.csv`

The report includes label counts, per-annotator totals, overlapping double-labeled frame counts, pairwise agreement, Cohen's kappa, disagreements, and legacy labels that are no longer in the current config.

## Google Drive / multi-Mac workflow

Recommended setup:

1. Keep the old MacBook as the always-on collector.
2. Let it store live capture data locally in `data/`.
3. Sync immutable image files to Google Drive if desired.
4. Do not put the live SQLite database directly inside a continuously synced Google Drive folder.
5. Instead, periodically create a safe database snapshot and sync that snapshot.

Example safe database backup:

```bash
envirocam backup-db \
  --config configs/mount_tam.yaml \
  --output "$HOME/Library/CloudStorage/GoogleDrive-YOURACCOUNT/My Drive/envirocam/mount_tam.sqlite3"
```

The exact Google Drive path may differ depending on how Google Drive for Desktop is installed. The key idea is: the collector writes the live DB locally, and Google Drive receives periodic backup copies.

## Current package layout

```text
src/enviro_webcam_ml/
  annotation.py       # local split-screen labeling web app
  annotation_analysis.py # annotation counts, agreement, disagreement reports
  backup.py           # safe SQLite snapshot backups
  capture.py          # webcam image fetch + immutable storage
  cli.py              # envirocam command line entry points
  config.py           # YAML config loading and validation
  dataset.py          # CSV manifest builder
  db.py               # SQLite schema and repository functions
  quality.py          # basic image quality heuristics
  weather/
    open_meteo.py     # Open-Meteo forecast adapter
```

## What this MVP does not do yet

It does not train a neural network yet. That is deliberate: the highest-leverage first step is creating trustworthy, timestamped, leakage-safe data. Once image/weather capture is stable, the next layer is annotation tooling and baseline models.
