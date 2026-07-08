from pathlib import Path

from PIL import Image

from enviro_webcam_ml.quality import assess_image_quality


def test_assess_image_quality_marks_dark_image_as_night(tmp_path: Path) -> None:
    path = tmp_path / "dark.jpg"
    Image.new("RGB", (16, 16), color=(5, 5, 5)).save(path)

    quality = assess_image_quality(
        path,
        previous_sha=None,
        current_sha="abc",
        night_luminance_threshold=35,
        blur_variance_threshold=25,
    )

    assert quality["is_night"] is True
    assert quality["avg_luminance"] < 35
