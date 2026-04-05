"""
step1_ingest/scanner.py
Scans the dataset folder structure, detects parts, annotations, and images on disk.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"})


class AnnotationFormat(str, Enum):
    COCO_JSON = "coco_json"
    CVAT_XML = "cvat_xml"


ANNOTATION_EXTENSIONS: dict[str, AnnotationFormat] = {
    ".json": AnnotationFormat.COCO_JSON,
    ".xml": AnnotationFormat.CVAT_XML,
}


@dataclass
class DatasetPart:
    name: str
    format: AnnotationFormat
    annotation_file: Path
    images_dir: Path
    image_files: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"DatasetPart(name={self.name!r}, format={self.format.value!r}, "
            f"images={len(self.image_files)})"
        )


def _detect_annotation_file(
        part_dir: Path
) -> tuple[Path, AnnotationFormat] | None:
    found: dict[str, Path] = {}

    for f in part_dir.iterdir():
        if f.is_file() and f.suffix.lower() in ANNOTATION_EXTENSIONS:
            ext = f.suffix.lower()
            if ext not in found:
                found[ext] = f

    if not found:
        logger.warning(
            "[scanner] Annotation file not found in %s. Supported extensions: %s",
            part_dir, list(ANNOTATION_EXTENSIONS.keys()))
        return None

    if len(found) > 1:
        logger.warning(
            "[scanner] Multiple annotation files found in %s: %s — using the first match.",
            part_dir,
            list(found.keys()),
        )

    for ext, fmt in ANNOTATION_EXTENSIONS.items():
        if ext in found:
            return found[ext], fmt

    return None


def _collect_images(images_dir: Path) -> list[str]:
    if not images_dir.exists():
        logger.warning("[scanner] Folder not found: %s", images_dir)
        return []
    if not images_dir.is_dir():
        logger.warning("[scanner] The path is not a directory: %s", images_dir)
        return []

    files = sorted(
        f.name
        for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    return files


class DatasetScanner:
    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise FileNotFoundError(f"dataset not found: {self.root_path}")
        if not self.root_path.is_dir():
            raise NotADirectoryError(f"path is not a folder: {self.root_path}")

    def scan(self) -> list[DatasetPart]:
        parts: list[DatasetPart] = []

        subdirs = sorted(
            d for d in self.root_path.iterdir() if d.is_dir()
        )

        if not subdirs:
            logger.warning("[scanner] No subfolders found in: %s",
                           self.root_path)
            return parts

        for part_dir in subdirs:
            result = _detect_annotation_file(part_dir)
            if result is None:
                continue

            annotation_file, fmt = result
            images_dir = part_dir / "data"
            image_files = _collect_images(images_dir)

            if not image_files:
                logger.warning(
                    "[scanner] Part '%s': the 'data' folder is empty or not found. Found annotation file: %s",
                    part_dir.name,
                )

            part = DatasetPart(
                name=part_dir.name,
                format=fmt,
                annotation_file=annotation_file,
                images_dir=images_dir,
                image_files=image_files,
            )
            parts.append(part)

            logger.info(
                "[scanner] Found: %s | format: %s | images on disk: %d",
                part.name,
                part.format.value,
                len(part.image_files),
            )

        return parts
