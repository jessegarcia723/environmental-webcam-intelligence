from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class ImageTrainingOptions:
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


def train_image_model(options: ImageTrainingOptions) -> dict[str, Any]:
    torch, nn, optim, DataLoader, datasets, models, transforms = import_torch_stack()

    rows = read_training_rows(options.training_csv)
    if not rows:
        raise ValueError(f"No training rows found in {options.training_csv}")

    labels = sorted({row["label"] for row in rows})
    if len(labels) < 2:
        raise ValueError("Need at least two labels to train an image classifier.")

    label_to_idx = {label: index for index, label in enumerate(labels)}
    split_counts = Counter(row["split"] for row in rows)
    if split_counts.get("train", 0) == 0:
        raise ValueError("Training CSV has no train rows.")

    torch.manual_seed(options.seed)
    device = choose_device(torch, options.device)
    transform_train, transform_eval = build_transforms(transforms, options.image_size)

    train_dataset = CsvImageDataset(rows, label_to_idx, split="train", transform=transform_train)
    val_dataset = CsvImageDataset(rows, label_to_idx, split="val", transform=transform_eval)
    test_dataset = CsvImageDataset(rows, label_to_idx, split="test", transform=transform_eval)

    train_loader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=True,
        num_workers=options.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=options.num_workers,
    ) if test_dataset else None

    model = build_model(models, nn, options.model_name, len(labels), options.pretrained)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=options.learning_rate)

    history = []
    best_val_accuracy = -1.0
    best_state = None

    for epoch in range(1, options.epochs + 1):
        train_metrics = run_epoch(
            torch,
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )
        val_metrics = (
            evaluate(torch, model, val_loader, criterion, device)
            if val_loader is not None
            else {"loss": None, "accuracy": None, "count": 0}
        )
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            }
        )
        val_accuracy = val_metrics["accuracy"]
        if val_accuracy is not None and val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = (
        evaluate(torch, model, test_loader, criterion, device)
        if test_loader is not None
        else {"loss": None, "accuracy": None, "count": 0}
    )

    options.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = options.output_dir / "model.pt"
    metadata_path = options.output_dir / "metadata.json"
    torch.save(
        {
            "model_name": options.model_name,
            "model_state_dict": model.state_dict(),
            "label_to_idx": label_to_idx,
            "image_size": options.image_size,
            "pretrained": options.pretrained,
        },
        checkpoint_path,
    )

    summary = {
        "training_csv": str(options.training_csv),
        "output_dir": str(options.output_dir),
        "checkpoint_path": str(checkpoint_path),
        "metadata_path": str(metadata_path),
        "model_name": options.model_name,
        "pretrained": options.pretrained,
        "device": str(device),
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "learning_rate": options.learning_rate,
        "labels": labels,
        "label_to_idx": label_to_idx,
        "split_counts": dict(sorted(split_counts.items())),
        "history": history,
        "test": test_metrics,
    }
    metadata_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


class CsvImageDataset:
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
        with Image.open(row["image_path"]) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, self.label_to_idx[row["label"]]


def read_training_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    usable = []
    for row in rows:
        if row.get("image_exists") not in ("1", "True", "true", "yes"):
            continue
        if not row.get("image_path") or not Path(row["image_path"]).exists():
            continue
        if not row.get("label") or not row.get("split"):
            continue
        usable.append(row)
    return usable


def build_model(models, nn, model_name: str, num_classes: int, pretrained: bool):
    if model_name != "resnet18":
        raise ValueError(f"Unsupported model_name={model_name!r}; currently supported: resnet18")

    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_transforms(transforms, image_size: int):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def choose_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_epoch(torch, model, loader, criterion, device, *, optimizer):
    model.train()
    total_loss = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_count = labels.size(0)
        total_loss += float(loss.detach().cpu()) * batch_count
        correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
        count += batch_count
    return metric_summary(total_loss, correct, count)


def evaluate(torch, model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_count = labels.size(0)
            total_loss += float(loss.detach().cpu()) * batch_count
            correct += int((torch.argmax(logits, dim=1) == labels).sum().detach().cpu())
            count += batch_count
    return metric_summary(total_loss, correct, count)


def metric_summary(total_loss: float, correct: int, count: int) -> dict[str, float | int | None]:
    if count == 0:
        return {"loss": None, "accuracy": None, "count": 0}
    return {
        "loss": total_loss / count,
        "accuracy": correct / count,
        "count": count,
    }


def import_torch_stack():
    try:
        import torch
        from torch import nn, optim
        from torch.utils.data import DataLoader
        from torchvision import datasets, models, transforms
    except ImportError as exc:
        raise RuntimeError(
            "Training requires PyTorch/TorchVision. Install with: "
            'python -m pip install -e ".[dev,train]" in a Python 3.10+ environment.'
        ) from exc
    return torch, nn, optim, DataLoader, datasets, models, transforms
