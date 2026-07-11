from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from enviro_webcam_ml.image_preprocessing import PixelCrop, crop_to_dict
from enviro_webcam_ml.image_training import (
    build_transforms,
    choose_device,
    class_weight_tensor,
    classification_metrics,
    decision_indexes,
    import_torch_stack,
    label_counts_by_split,
    metric_summary,
    read_training_rows,
    validate_class_weights,
)
from enviro_webcam_ml.weather_lasso import (
    assign_weather_hour_blocked_splits,
    build_weather_examples,
    inferred_feature_names,
    validate_blocked_fractions,
    validate_split_strategy,
    weather_group_leakage,
    weather_records_by_camera,
)


@dataclass(frozen=True)
class ImageWeatherTrainingOptions:
    training_csv: Path
    output_dir: Path
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 0.001
    image_size: int = 224
    num_workers: int = 0
    model_name: str = "resnet18"
    pretrained: bool = False
    device: str = "auto"
    seed: int = 42
    crop_pixels: PixelCrop | dict[str, int] | None = None
    positive_label: str | None = None
    positive_threshold: float | None = None
    class_weights: dict[str, float] | None = None
    weather_features: tuple[str, ...] = ()
    max_weather_age_minutes: float = 90.0
    split_strategy: str = "training_csv"
    blocked_val_fraction: float = 0.15
    blocked_test_fraction: float = 0.15
    weather_hidden_dim: int = 16
    fusion_hidden_dim: int = 128
    dropout: float = 0.2


def train_image_weather_model(
    conn: sqlite3.Connection,
    options: ImageWeatherTrainingOptions,
) -> dict[str, Any]:
    torch, nn, optim, DataLoader, _datasets, models, transforms = import_torch_stack()

    rows = read_training_rows(options.training_csv)
    if not rows:
        raise ValueError(f"No usable image training rows found in {options.training_csv}")
    validate_split_strategy(options.split_strategy)
    validate_blocked_fractions(options.blocked_val_fraction, options.blocked_test_fraction)

    labels = sorted({row["label"] for row in rows})
    if len(labels) < 2:
        raise ValueError("Need at least two labels to train an image+weather classifier.")
    if options.positive_label and options.positive_label not in labels:
        raise ValueError(f"positive_label={options.positive_label!r} is not present in training labels: {labels}")
    validate_class_weights(options.class_weights or {}, labels)
    if options.weather_hidden_dim <= 0:
        raise ValueError("--weather-hidden-dim must be > 0.")
    if options.fusion_hidden_dim <= 0:
        raise ValueError("--fusion-hidden-dim must be > 0.")
    if not 0 <= options.dropout < 1:
        raise ValueError("--dropout must be >= 0 and < 1.")

    weather = weather_records_by_camera(conn)
    examples, skipped = build_weather_examples(
        rows,
        weather,
        requested_features=options.weather_features,
        max_weather_age_minutes=options.max_weather_age_minutes,
        positive_label=options.positive_label or labels[0],
    )
    examples = [example for example in examples if example.get("image_path") and Path(example["image_path"]).exists()]
    if not examples:
        raise ValueError(
            "No image training rows could be matched to weather records. "
            "Run envirocam fetch-weather/run-collector, or increase --max-weather-age-minutes."
        )
    if options.split_strategy == "weather-hour-blocked":
        blocked_split_summary = assign_weather_hour_blocked_splits(
            examples,
            val_fraction=options.blocked_val_fraction,
            test_fraction=options.blocked_test_fraction,
        )
    else:
        blocked_split_summary = None

    weather_features = list(options.weather_features or inferred_feature_names(examples))
    if not weather_features:
        raise ValueError("No numeric weather features were found in matched weather records.")

    train_examples = [example for example in examples if example["split"] == "train"]
    if not train_examples:
        raise ValueError("No train rows with matched image+weather records.")

    weather_stats = weather_normalization_stats(train_examples, weather_features)
    label_to_idx = {label: index for index, label in enumerate(labels)}
    split_counts = Counter(example["split"] for example in examples)
    split_label_counts = label_counts_by_split(examples, labels)
    torch.manual_seed(options.seed)
    device = choose_device(torch, options.device)
    transform_train, transform_eval = build_transforms(
        transforms,
        options.image_size,
        crop_pixels=options.crop_pixels,
    )

    train_dataset = CsvImageWeatherDataset(
        examples,
        label_to_idx,
        weather_features,
        weather_stats,
        split="train",
        transform=transform_train,
        torch=torch,
    )
    val_dataset = CsvImageWeatherDataset(
        examples,
        label_to_idx,
        weather_features,
        weather_stats,
        split="val",
        transform=transform_eval,
        torch=torch,
    )
    test_dataset = CsvImageWeatherDataset(
        examples,
        label_to_idx,
        weather_features,
        weather_stats,
        split="test",
        transform=transform_eval,
        torch=torch,
    )
    train_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=True, num_workers=options.num_workers)
    train_eval_loader = DataLoader(train_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers) if val_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=options.batch_size, shuffle=False, num_workers=options.num_workers) if test_dataset else None

    image_encoder, image_feature_dim = build_image_encoder(models, nn, options.model_name, options.pretrained)
    model = ImageWeatherFusionModel(
        nn,
        image_encoder=image_encoder,
        image_feature_dim=image_feature_dim,
        weather_feature_dim=len(weather_features),
        num_classes=len(labels),
        weather_hidden_dim=options.weather_hidden_dim,
        fusion_hidden_dim=options.fusion_hidden_dim,
        dropout=options.dropout,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight_tensor(torch, options.class_weights or {}, labels, device))
    optimizer = optim.AdamW(model.parameters(), lr=options.learning_rate)

    history = []
    best_val_accuracy = -1.0
    best_state = None
    for epoch in range(1, options.epochs + 1):
        train_metrics = run_image_weather_epoch(
            torch,
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = (
            evaluate_image_weather(torch, model, val_loader, criterion, device)
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
        evaluate_image_weather(torch, model, test_loader, criterion, device)
        if test_loader is not None
        else {"loss": None, "accuracy": None, "count": 0}
    )
    prediction_rows = []
    for split, loader in (("train", train_eval_loader), ("val", val_loader), ("test", test_loader)):
        if loader is None:
            continue
        prediction_rows.extend(
            collect_image_weather_predictions(
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
            "model_type": "image_weather_fusion",
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "image_size": options.image_size,
            "pretrained": options.pretrained,
            "crop_pixels": crop_to_dict(options.crop_pixels),
            "positive_label": options.positive_label,
            "positive_threshold": options.positive_threshold,
            "class_weights": options.class_weights or {},
            "weather_features": weather_features,
            "weather_stats": weather_stats,
            "weather_hidden_dim": options.weather_hidden_dim,
            "fusion_hidden_dim": options.fusion_hidden_dim,
            "dropout": options.dropout,
        },
        checkpoint_path,
    )

    summary = {
        "training_csv": str(options.training_csv),
        "output_dir": str(options.output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "predictions_path": str(predictions_path),
        "model_name": options.model_name,
        "model_type": "image_weather_fusion",
        "pretrained": options.pretrained,
        "device": str(device),
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "learning_rate": options.learning_rate,
        "image_size": options.image_size,
        "crop_pixels": crop_to_dict(options.crop_pixels),
        "positive_label": options.positive_label,
        "positive_threshold": options.positive_threshold,
        "class_weights": options.class_weights or {},
        "labels": labels,
        "label_to_idx": label_to_idx,
        "split_strategy": options.split_strategy,
        "blocked_split_summary": blocked_split_summary,
        "weather_group_leakage": weather_group_leakage(examples),
        "max_weather_age_minutes": options.max_weather_age_minutes,
        "weather_features": weather_features,
        "weather_stats": weather_stats,
        "weather_hidden_dim": options.weather_hidden_dim,
        "fusion_hidden_dim": options.fusion_hidden_dim,
        "dropout": options.dropout,
        "split_counts": dict(sorted(split_counts.items())),
        "split_label_counts": split_label_counts,
        "matched_rows": len(examples),
        "skipped": skipped,
        "history": history,
        "test": test_metrics,
        "detailed_metrics": detailed_metrics,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_image_weather_predictions(prediction_rows, predictions_path)
    return summary


class ImageWeatherFusionModel:
    def __init__(
        self,
        nn,
        *,
        image_encoder,
        image_feature_dim: int,
        weather_feature_dim: int,
        num_classes: int,
        weather_hidden_dim: int,
        fusion_hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.module = nn.Module()
        self.module.image_encoder = image_encoder
        self.module.weather_encoder = nn.Sequential(
            nn.Linear(weather_feature_dim, weather_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.module.classifier = nn.Sequential(
            nn.Linear(image_feature_dim + weather_hidden_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def __call__(self, images, weather_vectors):
        image_features = self.module.image_encoder(images)
        if len(image_features.shape) > 2:
            image_features = image_features.flatten(start_dim=1)
        weather_features = self.module.weather_encoder(weather_vectors)
        import torch

        return self.module.classifier(torch.cat([image_features, weather_features], dim=1))

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


class CsvImageWeatherDataset:
    def __init__(
        self,
        rows,
        label_to_idx,
        weather_features: list[str],
        weather_stats: dict[str, dict[str, float]],
        *,
        split: str,
        transform,
        torch,
    ) -> None:
        self.rows = [row for row in rows if row["split"] == split]
        self.label_to_idx = label_to_idx
        self.weather_features = weather_features
        self.weather_stats = weather_stats
        self.transform = transform
        self.torch = torch

    def __len__(self) -> int:
        return len(self.rows)

    def __bool__(self) -> bool:
        return bool(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        weather_tensor = self.torch.tensor(
            normalized_weather_vector(row["features"], self.weather_features, self.weather_stats),
            dtype=self.torch.float32,
        )
        return image_tensor, weather_tensor, self.label_to_idx[row["label"]], index


def build_image_encoder(models, nn, model_name: str, pretrained: bool):
    if model_name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        return model, feature_dim
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    if model_name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        feature_dim = model.classifier[-1].in_features
        model.classifier = nn.Identity()
        return model, feature_dim
    supported = "resnet18, efficientnet_b0, mobilenet_v3_small"
    raise ValueError(f"Unsupported model_name={model_name!r}; currently supported: {supported}")


def weather_normalization_stats(
    examples: list[dict[str, Any]],
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    stats = {}
    for feature in feature_names:
        values = sorted(
            float(example["features"][feature])
            for example in examples
            if feature in example["features"]
        )
        if values:
            median = values[len(values) // 2]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = variance ** 0.5
        else:
            median = 0.0
            mean = 0.0
            std = 1.0
        stats[feature] = {
            "median": median,
            "mean": mean,
            "std": std if std > 0 else 1.0,
        }
    return stats


def normalized_weather_vector(
    features: dict[str, float],
    feature_names: list[str],
    stats: dict[str, dict[str, float]],
) -> list[float]:
    vector = []
    for feature in feature_names:
        feature_stats = stats[feature]
        value = features.get(feature, feature_stats["median"])
        vector.append((float(value) - feature_stats["mean"]) / feature_stats["std"])
    return vector


def run_image_weather_epoch(torch, model, loader, criterion, device, *, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, weather_vectors, labels, _indexes in loader:
        images = images.to(device)
        weather_vectors = weather_vectors.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images, weather_vectors)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_count
        correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
        count += batch_count
    return metric_summary(total_loss, correct, count)


def evaluate_image_weather(torch, model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, weather_vectors, labels, _indexes in loader:
            images = images.to(device)
            weather_vectors = weather_vectors.to(device)
            labels = labels.to(device)
            logits = model(images, weather_vectors)
            loss = criterion(logits, labels)
            batch_count = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_count
            correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
            count += batch_count
    return metric_summary(total_loss, correct, count)


def collect_image_weather_predictions(
    torch,
    model,
    loader,
    device,
    labels: list[str],
    split: str,
    *,
    positive_label: str | None = None,
    positive_threshold: float | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    rows = []
    positive_idx = labels.index(positive_label) if positive_label in labels else None
    with torch.no_grad():
        for images, weather_vectors, true_indexes, row_indexes in loader:
            images = images.to(device)
            weather_vectors = weather_vectors.to(device)
            logits = model(images, weather_vectors)
            probabilities = torch.softmax(logits, dim=1).detach().cpu()
            pred_indexes = decision_indexes(torch, probabilities, labels, positive_idx, positive_threshold)
            true_indexes = true_indexes.detach().cpu()
            row_indexes = row_indexes.detach().cpu()
            for batch_index in range(len(row_indexes)):
                source_row = loader.dataset.rows[int(row_indexes[batch_index])]
                true_idx = int(true_indexes[batch_index])
                pred_idx = int(pred_indexes[batch_index])
                confidence = float(probabilities[batch_index, pred_idx])
                positive_probability = (
                    float(probabilities[batch_index, positive_idx])
                    if positive_idx is not None
                    else ""
                )
                rows.append(
                    {
                        "split": split,
                        "capture_id": source_row.get("capture_id", ""),
                        "camera_id": source_row.get("camera_id", ""),
                        "captured_at_utc": source_row.get("captured_at_utc", ""),
                        "weather_valid_at_utc": source_row.get("weather_valid_at_utc", ""),
                        "weather_group": source_row.get("weather_group", ""),
                        "weather_age_minutes": source_row.get("weather_age_minutes", ""),
                        "true_label": labels[true_idx],
                        "pred_label": labels[pred_idx],
                        "confidence": confidence,
                        "positive_probability": positive_probability,
                        "correct": int(true_idx == pred_idx),
                        "image_path": source_row.get("image_path", ""),
                    }
                )
    return rows


def write_image_weather_predictions(predictions: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "split",
        "capture_id",
        "camera_id",
        "captured_at_utc",
        "weather_valid_at_utc",
        "weather_group",
        "weather_age_minutes",
        "true_label",
        "pred_label",
        "confidence",
        "positive_probability",
        "correct",
        "image_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)
