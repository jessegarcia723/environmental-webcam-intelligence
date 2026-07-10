from PIL import Image

from enviro_webcam_ml.image_preprocessing import crop_image, parse_pixel_crop


def test_parse_pixel_crop_supports_top_bottom_and_defaults() -> None:
    crop = parse_pixel_crop({"top": 160, "bottom": 31})

    assert crop is not None
    assert crop.as_dict() == {"top": 160, "bottom": 31, "left": 0, "right": 0}


def test_crop_image_removes_configured_edges() -> None:
    image = Image.new("RGB", (1920, 1080), "white")
    cropped = crop_image(image, {"top": 160, "bottom": 31})

    assert cropped.size == (1920, 889)


def test_empty_crop_returns_copy() -> None:
    image = Image.new("RGB", (10, 10), "white")
    cropped = crop_image(image, None)

    assert cropped.size == image.size
    assert cropped is not image
