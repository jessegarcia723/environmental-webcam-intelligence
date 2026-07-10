from __future__ import annotations

import csv
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from enviro_webcam_ml.image_training import (
    build_model,
    build_transforms,
    choose_device,
    import_torch_stack,
)


@dataclass(frozen=True)
class ImageExplanationOptions:
    checkpoint_path: Path
    predictions_path: Path
    output_dir: Path
    split: str = "test"
    selection: str = "mixed"
    max_images: int = 24
    target: str = "predicted"
    device: str = "auto"
    output_width: int = 960
    alpha: float = 0.45


def explain_image_model(options: ImageExplanationOptions) -> dict[str, Any]:
    if options.max_images <= 0:
        raise ValueError("--max-images must be > 0.")
    if options.output_width <= 0:
        raise ValueError("--output-width must be > 0.")
    if not 0.0 <= options.alpha <= 1.0:
        raise ValueError("--alpha must be between 0 and 1.")

    torch, nn, _optim, _DataLoader, _datasets, models, transforms = import_torch_stack()
    matplotlib = import_matplotlib()
    np = import_numpy()

    checkpoint = torch.load(options.checkpoint_path, map_location="cpu")
    label_to_idx = checkpoint.get("label_to_idx")
    if not label_to_idx:
        raise ValueError(f"Checkpoint is missing label_to_idx: {options.checkpoint_path}")
    labels = labels_from_mapping(label_to_idx)
    model_name = checkpoint.get("model_name", "resnet18")
    image_size = int(checkpoint.get("image_size", 224))

    device = choose_device(torch, options.device)
    model = build_model(models, nn, model_name, len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _train_transform, eval_transform = build_transforms(transforms, image_size)
    target_layer = grad_cam_target_layer(model, model_name)

    prediction_rows = read_prediction_rows(options.predictions_path)
    selected_rows = select_prediction_rows(
        prediction_rows,
        split=options.split,
        selection=options.selection,
        max_images=options.max_images,
    )
    if not selected_rows:
        raise ValueError(
            f"No usable prediction rows found in {options.predictions_path} "
            f"for split={options.split!r} selection={options.selection!r}."
        )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    runner = GradCamRunner(torch, model, target_layer, device)
    try:
        for index, row in enumerate(selected_rows, start=1):
            image_path = Path(row["image_path"])
            with Image.open(image_path) as original:
                original = original.convert("RGB")
                image_tensor = eval_transform(original)
                requested_target_idx = target_index_for_row(row, label_to_idx, options.target)
                cam, model_output = runner(image_tensor, requested_target_idx)
                output_image = render_gradcam_panel(
                    original,
                    cam,
                    matplotlib,
                    np,
                    output_width=options.output_width,
                    alpha=options.alpha,
                )

            output_name = explanation_filename(index, row)
            output_path = options.output_dir / output_name
            output_image.save(output_path, quality=92)
            generated.append(
                {
                    "file": output_name,
                    "capture_id": row.get("capture_id", ""),
                    "camera_id": row.get("camera_id", ""),
                    "captured_at_utc": row.get("captured_at_utc", ""),
                    "split": row.get("split", ""),
                    "true_label": row.get("true_label", ""),
                    "pred_label": row.get("pred_label", ""),
                    "target_label": labels[requested_target_idx],
                    "model_pred_label": labels[model_output["pred_idx"]],
                    "model_confidence": model_output["confidence"],
                    "prediction_csv_confidence": parse_float(row.get("confidence")),
                    "correct": row.get("correct", ""),
                    "image_path": str(image_path),
                }
            )
    finally:
        runner.close()

    index_path = options.output_dir / "index.html"
    summary_path = options.output_dir / "summary.json"
    write_explanation_index(index_path, generated, options, model_name, labels)
    summary = {
        "checkpoint_path": str(options.checkpoint_path),
        "predictions_path": str(options.predictions_path),
        "output_dir": str(options.output_dir),
        "index_path": str(index_path),
        "summary_path": str(summary_path),
        "model_name": model_name,
        "device": str(device),
        "labels": labels,
        "split": options.split,
        "selection": options.selection,
        "target": options.target,
        "count": len(generated),
        "explanations": generated,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


class GradCamRunner:
    def __init__(self, torch, model, target_layer, device) -> None:
        self.torch = torch
        self.model = model
        self.device = device
        self.activation = None
        self.handle = target_layer.register_forward_hook(self._capture_activation)

    def _capture_activation(self, _module, _inputs, output) -> None:
        if isinstance(output, tuple):
            output = output[0]
        output.retain_grad()
        self.activation = output

    def __call__(self, image_tensor, target_idx: int) -> tuple[Any, dict[str, Any]]:
        self.model.zero_grad(set_to_none=True)
        self.activation = None
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        logits = self.model(image_batch)
        probabilities = self.torch.softmax(logits, dim=1)
        pred_idx = int(self.torch.argmax(probabilities, dim=1).detach().cpu()[0])
        confidence = float(probabilities[0, pred_idx].detach().cpu())

        score = logits[0, target_idx]
        score.backward()
        if self.activation is None or self.activation.grad is None:
            raise RuntimeError("Grad-CAM failed to capture model activations/gradients.")

        activations = self.activation.detach()
        gradients = self.activation.grad.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = self.torch.relu((weights * activations).sum(dim=1))[0]
        cam = cam - cam.min()
        cam_max = cam.max()
        if float(cam_max.detach().cpu()) > 0:
            cam = cam / cam_max
        return cam.detach().cpu().numpy(), {"pred_idx": pred_idx, "confidence": confidence}

    def close(self) -> None:
        self.handle.remove()


def grad_cam_target_layer(model, model_name: str):
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name in {"efficientnet_b0", "mobilenet_v3_small"}:
        return model.features[-1]
    raise ValueError(f"No Grad-CAM target layer is configured for model_name={model_name!r}.")


def read_prediction_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    usable = []
    for row in rows:
        image_path = row.get("image_path", "")
        if not image_path or not Path(image_path).exists():
            continue
        usable.append(row)
    return usable


def select_prediction_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    selection: str,
    max_images: int,
) -> list[dict[str, str]]:
    if split != "all":
        rows = [row for row in rows if row.get("split") == split]

    correct_rows = [row for row in rows if is_correct(row)]
    incorrect_rows = [row for row in rows if not is_correct(row)]
    confidence_desc = lambda row: parse_float(row.get("confidence")) or 0.0
    confidence_asc = lambda row: parse_float(row.get("confidence")) if parse_float(row.get("confidence")) is not None else 2.0

    if selection == "incorrect":
        pool = sorted(incorrect_rows, key=confidence_desc, reverse=True)
    elif selection == "correct":
        pool = sorted(correct_rows, key=confidence_asc)
    elif selection == "high-confidence":
        pool = sorted(rows, key=confidence_desc, reverse=True)
    elif selection == "low-confidence":
        pool = sorted(rows, key=confidence_asc)
    elif selection == "mixed":
        pool = sorted(incorrect_rows, key=confidence_desc, reverse=True)
        seen = {id(row) for row in pool}
        for row in sorted(correct_rows, key=confidence_asc):
            if id(row) not in seen:
                pool.append(row)
    else:
        supported = "mixed, incorrect, correct, high-confidence, low-confidence"
        raise ValueError(f"Unsupported selection={selection!r}; choose one of: {supported}.")

    return pool[:max_images]


def target_index_for_row(row: dict[str, str], label_to_idx: dict[str, int], target: str) -> int:
    if target == "predicted":
        label = row.get("pred_label", "")
    elif target == "true":
        label = row.get("true_label", "")
    else:
        raise ValueError("--target must be 'predicted' or 'true'.")
    if label not in label_to_idx:
        raise ValueError(f"Label {label!r} is not present in the checkpoint labels.")
    return int(label_to_idx[label])


def render_gradcam_panel(
    original: Image.Image,
    cam,
    matplotlib,
    np,
    *,
    output_width: int,
    alpha: float,
) -> Image.Image:
    original_resized = resize_to_width(original, output_width)
    cmap = matplotlib.colormaps.get_cmap("turbo")
    heatmap_rgba = cmap(cam)
    heatmap_rgb = Image.fromarray((heatmap_rgba[:, :, :3] * 255).astype("uint8"))
    heatmap_rgb = heatmap_rgb.resize(original_resized.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(original_resized, heatmap_rgb, alpha=alpha)

    panel = Image.new("RGB", (original_resized.width * 2, original_resized.height), "white")
    panel.paste(original_resized, (0, 0))
    panel.paste(overlay, (original_resized.width, 0))
    return panel


def resize_to_width(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image.copy()
    height = max(1, round(image.height * (width / image.width)))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def write_explanation_index(
    path: Path,
    explanations: list[dict[str, Any]],
    options: ImageExplanationOptions,
    model_name: str,
    labels: list[str],
) -> None:
    cards = []
    for item in explanations:
        title = (
            f"{item['camera_id']} {item['captured_at_utc']} "
            f"true={item['true_label']} pred={item['pred_label']}"
        )
        metadata = [
            ("capture", item["capture_id"]),
            ("camera", item["camera_id"]),
            ("captured", item["captured_at_utc"]),
            ("split", item["split"]),
            ("true", item["true_label"]),
            ("pred", item["pred_label"]),
            ("target", item["target_label"]),
            ("confidence", f"{item['model_confidence']:.4f}"),
            ("correct", item["correct"]),
            ("image", item["image_path"]),
        ]
        details = "".join(
            f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>"
            for label, value in metadata
        )
        cards.append(
            "<article>"
            f"<h2>{html.escape(title)}</h2>"
            f"<a href=\"{html.escape(item['file'])}\">"
            f"<img src=\"{html.escape(item['file'])}\" alt=\"{html.escape(title)}\"></a>"
            f"<dl>{details}</dl>"
            "</article>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Grad-CAM explanations</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2rem; color: #17202a; }}
    header {{ max-width: 80rem; }}
    article {{ margin: 2rem 0; padding-bottom: 2rem; border-bottom: 1px solid #d5d8dc; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ccd1d1; }}
    dl {{ display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 0.25rem 1rem; }}
    dt {{ font-weight: 700; color: #566573; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    code {{ background: #f4f6f7; padding: 0.1rem 0.3rem; border-radius: 0.25rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Grad-CAM explanations</h1>
    <p>Each panel shows the original image on the left and a Grad-CAM heatmap overlay on the right.</p>
    <p>
      model=<code>{html.escape(model_name)}</code>,
      split=<code>{html.escape(options.split)}</code>,
      selection=<code>{html.escape(options.selection)}</code>,
      target=<code>{html.escape(options.target)}</code>,
      labels=<code>{html.escape(', '.join(labels))}</code>
    </p>
  </header>
  {''.join(cards)}
</body>
</html>
"""
    path.write_text(document, encoding="utf-8")


def explanation_filename(index: int, row: dict[str, str]) -> str:
    pieces = [
        f"{index:03d}",
        row.get("split", ""),
        row.get("camera_id", ""),
        row.get("capture_id", "") or Path(row.get("image_path", "image")).stem,
        f"true-{row.get('true_label', '')}",
        f"pred-{row.get('pred_label', '')}",
    ]
    return f"{safe_slug('_'.join(piece for piece in pieces if piece))}.jpg"


def labels_from_mapping(label_to_idx: dict[str, int]) -> list[str]:
    return [
        label
        for label, _idx in sorted(label_to_idx.items(), key=lambda item: int(item[1]))
    ]


def is_correct(row: dict[str, str]) -> bool:
    return str(row.get("correct", "")).strip().lower() in {"1", "true", "yes"}


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = re.sub(r"_+", "_", value).strip("_.")
    return value or "gradcam"


def import_matplotlib():
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError(
            "Grad-CAM explanations require matplotlib. Install with: "
            'python -m pip install -e ".[dev,train]" in a Python 3.10+ environment.'
        ) from exc
    return matplotlib


def import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Grad-CAM explanations require numpy. Install with: "
            'python -m pip install -e ".[dev,train]" in a Python 3.10+ environment.'
        ) from exc
    return np
