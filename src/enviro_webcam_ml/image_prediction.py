from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from enviro_webcam_ml.image_training import (
    build_model,
    build_transforms,
    choose_device,
    decision_indexes,
    import_torch_stack,
)


def predict_image_paths(
    checkpoint_path: Path,
    image_paths_by_capture: dict[int, str],
    *,
    device: str = "auto",
) -> dict[int, dict[str, Any]]:
    if not image_paths_by_capture:
        return {}

    torch, nn, _optim, _DataLoader, _datasets, models, transforms = import_torch_stack()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_to_idx = checkpoint.get("label_to_idx")
    if not label_to_idx:
        raise ValueError(f"Checkpoint is missing label_to_idx: {checkpoint_path}")

    labels = labels_from_mapping(label_to_idx)
    model_name = checkpoint.get("model_name", "resnet18")
    image_size = int(checkpoint.get("image_size", 224))
    crop_pixels = checkpoint.get("crop_pixels")
    positive_label = checkpoint.get("positive_label")
    positive_threshold = checkpoint.get("positive_threshold")
    positive_idx = labels.index(positive_label) if positive_label in labels else None

    selected_device = choose_device(torch, device)
    model = build_model(models, nn, model_name, len(labels), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(selected_device)
    model.eval()
    _train_transform, eval_transform = build_transforms(
        transforms,
        image_size,
        crop_pixels=crop_pixels,
    )

    predictions: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for capture_id, image_path in sorted(image_paths_by_capture.items()):
            path = Path(image_path)
            if not path.exists():
                continue
            with Image.open(path) as image:
                image = image.convert("RGB")
                image_tensor = eval_transform(image).unsqueeze(0).to(selected_device)
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()[0]
            pred_idx = int(
                decision_indexes(
                    torch,
                    probabilities.unsqueeze(0),
                    labels,
                    positive_idx,
                    positive_threshold,
                )[0]
            )
            predictions[capture_id] = {
                "split": "adjudication_live",
                "true_label": "",
                "pred_label": labels[pred_idx],
                "confidence": float(probabilities[pred_idx]),
                "positive_probability": float(probabilities[positive_idx]) if positive_idx is not None else "",
                "correct": "",
            }
    return predictions


def labels_from_mapping(label_to_idx: dict[str, int]) -> list[str]:
    return [
        label
        for label, _idx in sorted(label_to_idx.items(), key=lambda item: int(item[1]))
    ]
