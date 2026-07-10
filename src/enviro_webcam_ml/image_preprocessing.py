from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PixelCrop:
    top: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "top": self.top,
            "bottom": self.bottom,
            "left": self.left,
            "right": self.right,
        }

    def is_empty(self) -> bool:
        return self.top == 0 and self.bottom == 0 and self.left == 0 and self.right == 0


def parse_pixel_crop(raw: Any) -> PixelCrop | None:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ValueError("image_preprocessing.crop_pixels must be a mapping.")

    crop = PixelCrop(
        top=non_negative_int(raw.get("top", raw.get("top_px", 0)), "crop_pixels.top"),
        bottom=non_negative_int(raw.get("bottom", raw.get("bottom_px", 0)), "crop_pixels.bottom"),
        left=non_negative_int(raw.get("left", raw.get("left_px", 0)), "crop_pixels.left"),
        right=non_negative_int(raw.get("right", raw.get("right_px", 0)), "crop_pixels.right"),
    )
    return None if crop.is_empty() else crop


def crop_image(image: Image.Image, crop: PixelCrop | dict[str, int] | None) -> Image.Image:
    crop = normalize_crop(crop)
    if crop is None:
        return image.copy()

    width, height = image.size
    left = crop.left
    upper = crop.top
    right = width - crop.right
    lower = height - crop.bottom
    if left >= right or upper >= lower:
        raise ValueError(
            "Configured image crop removes the whole image: "
            f"image_size={width}x{height}, crop={crop.as_dict()}"
        )
    return image.crop((left, upper, right, lower))


def normalize_crop(crop: PixelCrop | dict[str, int] | None) -> PixelCrop | None:
    if crop is None:
        return None
    if isinstance(crop, PixelCrop):
        return None if crop.is_empty() else crop
    return parse_pixel_crop(crop)


def crop_to_dict(crop: PixelCrop | dict[str, int] | None) -> dict[str, int] | None:
    crop = normalize_crop(crop)
    return crop.as_dict() if crop is not None else None


def non_negative_int(value: Any, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative integer.") from exc
    if parsed < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return parsed
