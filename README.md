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
envirocam check-training-env
envirocam build-training-set --config configs/mount_tam_training.yaml
envirocam train-image-model
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

## Training setup on the M5 MacBook

Use a newer Python on the M5 for training, preferably Python 3.11 or 3.12. The old collector can stay on Python 3.9; it does not need PyTorch.

Clone/update the repo and install training dependencies:

```bash
cd ~/Documents/environmental-webcam-intelligence
git pull

python3.12 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,train]"
```

If your Python command is `python3.11`, use that instead of `python3.12`.

Point the training config at the Google Drive-synced data folder:

```bash
export ENVIROCAM_DATA_DIR="/Users/jessegarica/Library/CloudStorage/GoogleDrive-jessegarcia723@gmail.com/Other computers/My MacBook Pro/environmental-webcam-intelligence/data"
```

Verify the synced database/config can be read:

```bash
envirocam analyze-annotations --config configs/mount_tam_training.yaml
```

Check installed ML packages and Apple Silicon acceleration:

```bash
envirocam check-training-env
```

On Apple Silicon, a good result will show `torch` installed and `mps_available: True`. That means PyTorch can use Apple's Metal backend.

Build the first clean image-training CSV:

```bash
envirocam build-training-set --config configs/mount_tam_training.yaml
```

By default this includes only captures where at least two annotators agree, skips disagreements, skips legacy labels, and excludes `night_unusable` and `camera_artifact`. It writes:

```text
data/training/marine_layer_detection_training.csv
```

The training-set builder also remaps old absolute image paths stored by the collector Mac to the configured `data_dir`, so a database synced from the old Mac can still point at images under the M5's Google Drive path.

Train the first image-only model:

```bash
envirocam train-image-model \
  --config configs/mount_tam_training.yaml \
  --epochs 5 \
  --model-name resnet18 \
  --device mps
```

This trains an image classifier using PyTorch. By default, `--device auto` uses `mps` on Apple Silicon when available, then CUDA, then CPU. Passing `--device mps` makes the Apple Silicon choice explicit. It writes:

```text
data/models/marine_layer_detection/resnet18/model.pt
data/models/marine_layer_detection/resnet18/metadata.json
data/models/marine_layer_detection/resnet18/predictions.csv
```

Supported model names:

- `resnet18`
- `efficientnet_b0`
- `mobilenet_v3_small`

Recommended comparison run:

```bash
for model in resnet18 efficientnet_b0 mobilenet_v3_small; do
  envirocam train-image-model \
    --config configs/mount_tam_training.yaml \
    --epochs 8 \
    --model-name "$model" \
    --pretrained \
    --device mps
done

envirocam compare-image-models --config configs/mount_tam_training.yaml
```

Each model's `metadata.json` includes overall, per-label, and per-camera metrics. Each `predictions.csv` includes one row per evaluated image, with `camera_id`, true label, predicted label, confidence, and correctness. Camera comparison groups come from the task config, so this works for other sites/scenarios too.

The comparison command writes:

```text
data/models/marine_layer_detection/comparison.csv
data/models/marine_layer_detection/comparison.md
```

My current hunch: with a small dataset, a pretrained `efficientnet_b0` or `mobilenet_v3_small` will likely beat a non-pretrained `resnet18`. But we should trust the held-out test metrics, not the hunch.

Generate Grad-CAM visual explanations for a trained image model:

```bash
envirocam explain-image-model \
  --config configs/mount_tam_training.yaml \
  --model-name efficientnet_b0 \
  --split test \
  --selection mixed \
  --max-images 30 \
  --device mps
```

This writes side-by-side original/heatmap panels plus an HTML gallery:

```text
data/models/marine_layer_detection/efficientnet_b0/explanations/index.html
data/models/marine_layer_detection/efficientnet_b0/explanations/summary.json
```

Open the gallery with:

```bash
open data/models/marine_layer_detection/efficientnet_b0/explanations/index.html
```

Useful selection modes are `mixed`, `incorrect`, `low-confidence`, `high-confidence`, and `correct`. `mixed` is the best first look: it shows mistakes first, then lower-confidence correct predictions if there are no mistakes. The heatmap highlights image regions that most influenced the selected class; it is a sanity check, not a causal proof.

## Generalizing to a new geophysical scenario

Scenario-specific values belong in YAML config, not framework code. A new site/task should define:

- `project.name`, `database_path`, and `data_dir`
- one or more `cameras`
- weather provider settings
- one or more `tasks`

Task config supports:

```yaml
tasks:
  - id: example_detection
    default: true
    output_slug: example_detection
    training_csv: ../data/training/example_detection_training.csv
    model_dir: ../data/models/example_detection
    labels:
      - positive
      - negative
      - uncertain
      - bad_frame
    excluded_training_labels:
      - uncertain
      - bad_frame
    comparison_groups:
      camera:
        - camera_a
        - camera_b
```

Most commands can then omit `--task-id` and output paths; they read defaults from the config.

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
  image_training.py   # PyTorch image classifier training
  image_explanations.py # Grad-CAM visual model explanations
  model_comparison.py # compare trained image-model runs
  training_dataset.py # agreed-label CSV builder for model training
  training_env.py     # ML package and accelerator environment checks
  weather/
    open_meteo.py     # Open-Meteo forecast adapter
```

## What this MVP does not do yet

It does not yet include weather-feature models, deployment serving, or forecast-horizon models. The current training command is an image-only baseline.
