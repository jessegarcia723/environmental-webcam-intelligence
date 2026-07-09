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

For two Bluetooth Xbox controllers on a MacBook, pair both controllers in macOS Bluetooth settings before opening the app. Chrome is the recommended browser because its Gamepad API support is the most reliable on macOS. The first connected controller controls the left pane; the second connected controller controls the right pane.

Annotations are saved into `data/mount_tam.sqlite3` in the `annotation` table.

## Current package layout

```text
src/enviro_webcam_ml/
  annotation.py       # local split-screen labeling web app
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
