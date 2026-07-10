import json
from pathlib import Path

from enviro_webcam_ml.model_comparison import compare_image_models


def test_compare_image_models_writes_ranked_outputs(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    write_metadata(models_dir / "resnet18" / "metadata.json", "resnet18", 0.5)
    write_metadata(models_dir / "efficientnet_b0" / "metadata.json", "efficientnet_b0", 0.75)
    output_csv = models_dir / "comparison.csv"
    output_md = models_dir / "comparison.md"

    rows = compare_image_models(models_dir, output_csv, output_md)

    assert rows[0]["run"] == "efficientnet_b0"
    assert rows[0]["test_accuracy"] == 0.75
    assert output_csv.exists()
    assert "efficientnet_b0" in output_md.read_text(encoding="utf-8")


def write_metadata(path: Path, model_name: str, test_accuracy: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "pretrained": True,
                "device": "mps",
                "epochs": 3,
                "split_counts": {"train": 10, "val": 4, "test": 4},
                "predictions_path": str(path.parent / "predictions.csv"),
                "detailed_metrics": {
                    "val": {
                        "overall": {"accuracy": 0.5, "count": 4},
                        "by_camera": {},
                    },
                    "test": {
                        "overall": {"accuracy": test_accuracy, "count": 4},
                        "by_camera": {
                            "mount_tam_east_peak": {"accuracy": test_accuracy, "count": 2},
                            "mount_tam_west_peak": {"accuracy": test_accuracy, "count": 2},
                        },
                    },
                },
            }
        ),
        encoding="utf-8",
    )
