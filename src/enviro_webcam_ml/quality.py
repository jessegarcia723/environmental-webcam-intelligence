from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageStat


def assess_image_quality(
    path: Path,
    *,
    previous_sha: str | None,
    current_sha: str,
    night_luminance_threshold: float,
    blur_variance_threshold: float,
) -> dict[str, Any]:
    with Image.open(path) as img:
        gray = img.convert("L")
        avg_luminance = float(ImageStat.Stat(gray).mean[0])
        blur_variance = gradient_variance(gray)

    is_duplicate = bool(previous_sha and previous_sha == current_sha)
    is_night = avg_luminance < night_luminance_threshold
    is_blurry = blur_variance < blur_variance_threshold

    flags: dict[str, Any] = {}
    if is_duplicate:
        flags["duplicate_or_frozen"] = True
    if is_night:
        flags["night_or_underexposed"] = True
    if is_blurry:
        flags["possibly_blurry"] = True

    return {
        "avg_luminance": avg_luminance,
        "blur_variance": blur_variance,
        "is_night": is_night,
        "is_blurry": is_blurry,
        "is_duplicate": is_duplicate,
        "flags": flags,
    }


def gradient_variance(gray: Image.Image) -> float:
    """Cheap sharpness proxy based on adjacent-pixel differences."""
    shifted_x = ImageChops.offset(gray, 1, 0)
    shifted_y = ImageChops.offset(gray, 0, 1)
    diff_x = ImageChops.difference(gray, shifted_x)
    diff_y = ImageChops.difference(gray, shifted_y)
    stat_x = ImageStat.Stat(diff_x)
    stat_y = ImageStat.Stat(diff_y)
    return float(stat_x.var[0] + stat_y.var[0])
