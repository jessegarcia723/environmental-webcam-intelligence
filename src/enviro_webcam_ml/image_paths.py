from __future__ import annotations

from pathlib import Path


def resolve_image_path(stored_path: str | None, data_dir: Path) -> Path | None:
    """Resolve an image path stored by another machine against this machine's data dir."""
    if not stored_path:
        return None
    path = Path(stored_path)
    if path.exists():
        return path
    if not path.is_absolute():
        return (data_dir / path).resolve()

    parts = path.parts
    data_indexes = [index for index, part in enumerate(parts) if part == "data"]
    if data_indexes:
        suffix = Path(*parts[data_indexes[-1] + 1 :])
        return (data_dir / suffix).resolve()
    return path
