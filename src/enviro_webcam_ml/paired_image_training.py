from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from enviro_webcam_ml.image_preprocessing import PixelCrop, crop_to_dict
from enviro_webcam_ml.image_training import (
    build_transforms,
    choose_device,
    classification_metrics,
    decision_indexes,
    import_torch_stack,
    metric_summary,
)
from enviro_webcam_ml.image_weather_training import build_image_encoder
from enviro_webcam_ml.paired_events import NEGATIVE_EVENT_LABEL, POSITIVE_EVENT_LABEL
from enviro_webcam_ml.weather_lasso import chronological_split_names, parse_datetime


@dataclass(frozen=True)
class PairedImageTrainingOptions:
    paired_events_csv: Path
    output_dir: Path
    camera_ids: tuple[str, str]
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    image_size: int = 224
    num_workers: int = 0
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    device: str = "auto"
    seed: int = 42
    crop_pixels: PixelCrop | dict[str, int] | None = None
    positive_label: str = POSITIVE_EVENT_LABEL
    positive_threshold: float | None = None
    split_strategy: str = "event-hour-blocked"
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    fusion_hidden_dim: int = 128
    dropout: float = 0.2


def train_paired_image_model(options: PairedImageTrainingOptions) -> dict[str, Any]:
    if len(options.camera_ids) != 2:
        raise ValueError("Paired image training requires exactly two camera IDs.")
    if options.fusion_hidden_dim <= 0:
        raise ValueError("--fusion-hidden-dim must be > 0.")
    if not 0 <= options.dropout < 1:
        raise ValueError("--dropout must be >= 0 and < 1.")
    validate_paired_image_split_strategy(options.split_strategy)

    torch, nn, optim, DataLoader, _datasets, models, transforms = import_torch_stack()
    rows = read_paired_image_rows(options.paired_events_csv, camera_ids=options.camera_ids)
    if not rows:
        raise ValueError(f"No usable paired image rows found in {options.paired_events_csv}")
    labels = [NEGATIVE_EVENT_LABEL, options.positive_label]
    if options.split_strategy == "event-hour-blocked":
        split_summary = assign_paired_image_event_hour_splits(
            rows,
            positive_label=options.positive_label,
            train_fraction=options.train_fraction,
            val_fraction=options.val_fraction,
        )
    else:
        split_summary = assign_paired_image_splits(
            rows,
            positive_label=options.positive_label,
            train_fraction=options.train_fraction,
            val_fraction=options.val_fraction,
        )
    label_to_idx = {label: index for index, label in enumerate(labels)}
    split_counts = Counter(row["split"] for row in rows)
    if split_counts.get("train", 0) == 0:
        raise ValueError("No train paired image rows available.")

    torch.manual_seed(options.seed)
    device = choose_device(torch, options.device)
    transform_train, transform_eval = build_transforms(
        transforms,
        options.image_size,
        crop_pixels=options.crop_pixels,
    )
    train_dataset = PairedImageDataset(rows, label_to_idx, split="train", transform=transform_train)
    train_eval_dataset = PairedImageDataset(rows, label_to_idx, split="train", transform=transform_eval)
    val_dataset = PairedImageDataset(rows, label_to_idx, split="val", transform=transform_eval)
    test_dataset = PairedImageDataset(rows, label_to_idx, split="test", transform=transform_eval)

    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=options.num_workers)
    train_eval_loader = DataLoader(train_eval_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers) if test_dataset else None

    image_encoder, image_feature_dim = build_image_encoder(models, nn, options.model_name, options.pretrained)
    model = PairedImageFusionModel(
        nn,
        image_encoder=image_encoder,
        image_feature_dim=image_feature_dim,
        num_classes=len(labels),
        fusion_hidden_dim=options.fusion_hidden_dim,
        dropout=options.dropout,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=options.learning_rate)
    history = []
    best_val_accuracy = -1.0
    best_state = None
    for epoch in range(1, options.epochs + 1):
        train_metrics = run_paired_image_epoch(torch, model, train_loader, criterion, device, optimizer=optimizer)
        val_metrics = (
            evaluate_paired_image(torch, model, val_loader, criterion, device)
            if val_loader is not None
            else {"loss": None, "accuracy": None, "count": 0}
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        val_accuracy = val_metrics["accuracy"]
        if val_accuracy is not None and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = (
        evaluate_paired_image(torch, model, test_loader, criterion, device)
        if test_loader is not None
        else {"loss": None, "accuracy": None, "count": 0}
    )
    prediction_rows = []
    for split, loader in (("train", train_eval_loader), ("val", val_loader), ("test", test_loader)):
        if loader is None:
            continue
        prediction_rows.extend(
            collect_paired_image_predictions(
                torch,
                model,
                loader,
                device,
                labels,
                split,
                positive_label=options.positive_label,
                positive_threshold=options.positive_threshold,
            )
        )
    detailed_metrics = {
        split: classification_metrics(
            [row for row in prediction_rows if row["split"] == split],
            labels,
            positive_label=options.positive_label,
        )
        for split in ("train", "val", "test")
    }

    options.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = options.output_dir / "model.pt"
    metadata_path = options.output_dir / "metadata.json"
    predictions_path = options.output_dir / "predictions.csv"
    torch.save(
        {
            "model_name": options.model_name,
            "model_type": "paired_image_fusion",
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "image_size": options.image_size,
            "pretrained": options.pretrained,
            "crop_pixels": crop_to_dict(options.crop_pixels),
            "camera_ids": options.camera_ids,
            "positive_label": options.positive_label,
            "positive_threshold": options.positive_threshold,
            "fusion_hidden_dim": options.fusion_hidden_dim,
            "dropout": options.dropout,
        },
        checkpoint_path,
    )
    split_label_counts = {
        split: {
            label: sum(1 for row in rows if row["split"] == split and row["label"] == label)
            for label in labels
        }
        for split in ("train", "val", "test")
    }
    summary = {
        "paired_events_csv": str(options.paired_events_csv),
        "output_dir": str(options.output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "model_name": options.model_name,
        "model_type": "paired_image_fusion",
        "event_scope": "paired_event",
        "pretrained": options.pretrained,
        "device": str(device),
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "learning_rate": options.learning_rate,
        "image_size": options.image_size,
        "crop_pixels": crop_to_dict(options.crop_pixels),
        "camera_ids": list(options.camera_ids),
        "positive_label": options.positive_label,
        "positive_threshold": options.positive_threshold,
        "labels": labels,
        "label_to_idx": label_to_idx,
        "split_strategy": options.split_strategy,
        "blocked_split_summary": split_summary,
        "split_counts": dict(sorted(split_counts.items())),
        "split_label_counts": split_label_counts,
        "matched_rows": len(rows),
        "history": history,
        "test": test_metrics,
        "detailed_metrics": detailed_metrics,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_paired_image_predictions(prediction_rows, predictions_path)
    return summary


class PairedImageFusionModel:
    def __init__(
        self,
        nn,
        *,
        image_encoder,
        image_feature_dim: int,
        num_classes: int,
        fusion_hidden_dim: int,
        dropout: float,
    ) -> None:
        self.module = nn.Module()
        self.module.image_encoder = image_encoder
        self.module.classifier = nn.Sequential(
            nn.Linear(image_feature_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def __call__(self, image_a, image_b):
        import torch

        features_a = self.module.image_encoder(image_a)
        features_b = self.module.image_encoder(image_b)
        if len(features_a.shape) > 2:
            features_a = features_a.flatten(start_dim=1)
        if len(features_b.shape) > 2:
            features_b = features_b.flatten(start_dim=1)
        return self.module.classifier(torch.cat([features_a, features_b], dim=1))

    def to(self, device):
        self.module.to(device)
        return self

    def train(self):
        self.module.train()

    def eval(self):
        self.module.eval()

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state):
        return self.module.load_state_dict(state)

    def parameters(self):
        return self.module.parameters()


class PairedImageDataset:
    def __init__(self, rows, label_to_idx, *, split: str, transform) -> None:
        self.rows = [row for row in rows if row["split"] == split]
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return bool(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path_a"]) as image:
            image_a = self.transform(image.convert("RGB"))
        with Image.open(row["image_path_b"]) as image:
            image_b = self.transform(image.convert("RGB"))
        return image_a, image_b, self.label_to_idx[row["label"]], index


def read_paired_image_rows(path: Path, *, camera_ids: tuple[str, str]) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    result = []
    camera_a, camera_b = camera_ids
    for row in rows:
        image_path_a = row.get(f"{camera_a}_image_path") or ""
        image_path_b = row.get(f"{camera_b}_image_path") or ""
        if not image_path_a or not image_path_b:
            continue
        if not Path(image_path_a).exists() or not Path(image_path_b).exists():
            continue
        label = row.get("event_label") or ""
        if label != POSITIVE_EVENT_LABEL:
            label = NEGATIVE_EVENT_LABEL
        result.append(
            {
                "event_id": row.get("event_id", ""),
                "event_time_utc": row.get("event_time_utc", ""),
                "event_time_local": row.get("event_time_local", ""),
                "event_time_local_label": row.get("event_time_local_label", ""),
                "label": label,
                "image_path_a": image_path_a,
                "image_path_b": image_path_b,
                "label_pair": row.get("label_pair", ""),
                "split": row.get("split") or "",
            }
        )
    return result


def validate_paired_image_split_strategy(strategy: str) -> None:
    valid = {"event-hour-blocked", "chronological"}
    if strategy not in valid:
        raise ValueError(f"Unsupported paired image split strategy {strategy!r}; choose one of {sorted(valid)}.")


def assign_paired_image_splits(
    rows: list[dict[str, Any]],
    *,
    positive_label: str,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    split_by_id = {}
    for label in (NEGATIVE_EVENT_LABEL, positive_label):
        label_rows = [row for row in rows if row["label"] == label]
        label_rows.sort(key=lambda row: (row["event_time_utc"], row["event_id"]))
        splits = chronological_split_names(len(label_rows), train_fraction=train_fraction, val_fraction=val_fraction)
        for row, split in zip(label_rows, splits):
            split_by_id[row["event_id"]] = split
    for row in rows:
        row["original_split"] = row.get("split", "")
        row["split"] = split_by_id[row["event_id"]]
    return {
        "group_count": None,
        "row_split_counts": dict(Counter(row["split"] for row in rows)),
    }


def assign_paired_image_event_hour_splits(
    rows: list[dict[str, Any]],
    *,
    positive_label: str,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(event_hour_group_key(row), []).append(row)

    group_rows = []
    for key, group in groups.items():
        first = min(group, key=lambda row: (row["event_time_utc"], row["event_id"]))
        group_rows.append(
            {
                "group": key,
                "first_event_time_utc": first["event_time_utc"],
                "target": int(any(row["label"] == positive_label for row in group)),
                "row_count": len(group),
            }
        )

    split_by_group: dict[str, str] = {}
    for target in (0, 1):
        target_groups = [group for group in group_rows if group["target"] == target]
        target_groups.sort(key=lambda group: (group["first_event_time_utc"], group["group"]))
        splits = chronological_split_names(
            len(target_groups),
            train_fraction=train_fraction,
            val_fraction=val_fraction,
        )
        for group, split in zip(target_groups, splits):
            split_by_group[group["group"]] = split

    for row in rows:
        row["original_split"] = row.get("split", "")
        row["split"] = split_by_group[event_hour_group_key(row)]

    return {
        "group_count": len(group_rows),
        "group_split_counts": dict(Counter(split_by_group.values())),
        "row_split_counts": dict(Counter(row["split"] for row in rows)),
        "positive_group_count": sum(1 for group in group_rows if group["target"]),
        "non_positive_group_count": sum(1 for group in group_rows if not group["target"]),
    }


def event_hour_group_key(row: dict[str, Any]) -> str:
    parsed = parse_datetime(row.get("event_time_utc", ""))
    return parsed.replace(minute=0, second=0, microsecond=0).isoformat()


def run_paired_image_epoch(torch, model, loader, criterion, device, *, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    for image_a, image_b, labels, _indexes in loader:
        image_a = image_a.to(device)
        image_b = image_b.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(image_a, image_b)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_count
        correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
        count += batch_count
    return metric_summary(total_loss, correct, count)


def evaluate_paired_image(torch, model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for image_a, image_b, labels, _indexes in loader:
            image_a = image_a.to(device)
            image_b = image_b.to(device)
            labels = labels.to(device)
            logits = model(image_a, image_b)
            loss = criterion(logits, labels)
            batch_count = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_count
            correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
            count += batch_count
    return metric_summary(total_loss, correct, count)


def collect_paired_image_predictions(
    torch,
    model,
    loader,
    device,
    labels: list[str],
    split: str,
    *,
    positive_label: str,
    positive_threshold: float | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    positive_idx = labels.index(positive_label)
    with torch.no_grad():
        for image_a, image_b, true_indexes, row_indexes in loader:
            image_a = image_a.to(device)
            image_b = image_b.to(device)
            logits = model(image_a, image_b)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            pred_indexes = decision_indexes(torch, probabilities, labels, positive_idx, positive_threshold)
            true_indexes = true_indexes.detach().cpu()
            row_indexes = row_indexes.detach().cpu()
            for batch_index in range(len(row_indexes)):
                source_row = loader.dataset.rows[int(row_indexes[batch_index])]
                true_idx = int(true_indexes[batch_index])
                pred_idx = int(pred_indexes[batch_index])
                rows.append(
                    {
                        "split": split,
                        "event_id": source_row.get("event_id", ""),
                        "event_time_utc": source_row.get("event_time_utc", ""),
                        "event_time_local": source_row.get("event_time_local", ""),
                        "event_time_local_label": source_row.get("event_time_local_label", ""),
                        "camera_id": "paired",
                        "true_label": labels[true_idx],
                        "pred_label": labels[pred_idx],
                        "confidence": float(probabilities[batch_index, pred_idx]),
                        "positive_probability": float(probabilities[batch_index, positive_idx]),
                        "correct": int(true_idx == pred_idx),
                        "image_path_a": source_row.get("image_path_a", ""),
                        "image_path_b": source_row.get("image_path_b", ""),
                    }
                )
    return rows


def write_paired_image_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "split",
        "event_id",
        "event_time_utc",
        "event_time_local",
        "event_time_local_label",
        "camera_id",
        "true_label",
        "pred_label",
        "confidence",
        "positive_probability",
        "correct",
        "image_path_a",
        "image_path_b",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
