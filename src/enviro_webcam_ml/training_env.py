from __future__ import annotations

import importlib
import importlib.metadata
import platform
import sys
from typing import Any


PACKAGES = [
    "numpy",
    "pandas",
    "sklearn",
    "matplotlib",
    "torch",
    "torchvision",
]


def training_environment_report() -> dict[str, Any]:
    packages = {name: package_version(name) for name in PACKAGES}
    torch_info = torch_capabilities()
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch": torch_info,
    }


def package_version(import_name: str) -> str | None:
    metadata_name = "scikit-learn" if import_name == "sklearn" else import_name
    try:
        return importlib.metadata.version(metadata_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def torch_capabilities() -> dict[str, Any]:
    if package_version("torch") is None:
        return {
            "installed": False,
            "mps_available": False,
            "cuda_available": False,
            "recommended_device": None,
        }

    torch = importlib.import_module("torch")
    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    cuda_available = bool(torch.cuda.is_available())
    if mps_available:
        recommended_device = "mps"
    elif cuda_available:
        recommended_device = "cuda"
    else:
        recommended_device = "cpu"
    return {
        "installed": True,
        "mps_available": mps_available,
        "cuda_available": cuda_available,
        "recommended_device": recommended_device,
    }
