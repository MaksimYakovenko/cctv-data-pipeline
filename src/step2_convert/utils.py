"""
utils.py — Step 2 Convert
Helper functions for writing YOLO label files and resolving image paths.
"""

from pathlib import Path


def to_yolo_line(class_id: int, cx: float, cy: float, w: float, h: float) -> str:
    """Formats a single YOLO line: 'class_id cx cy w h'."""
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def write_label_file(label_path: Path, lines: list[str]) -> None:
    """Writes a list of YOLO strings to a .txt file."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def find_actual_image(
        annotated_name: str,
        images_dir: Path,
        orphan_files: list[str],
) -> Path | None:
    """
    Returns the real path to the image:
    - first searches for an exact match by name in images_dir
    - if not found — looks for a file with the same stem in orphan_files (extension mismatch)
    """
    exact = images_dir / annotated_name
    if exact.exists():
        return exact

    stem = Path(annotated_name).stem
    for orphan in orphan_files:
        if Path(orphan).stem == stem:
            candidate = images_dir / orphan
            if candidate.exists():
                return candidate

    return None
