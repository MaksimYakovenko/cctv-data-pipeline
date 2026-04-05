"""
json_converter.py — Step 2 Convert
Converts COCO JSON annotations to YOLO format.
"""

import json
import logging
import shutil
from pathlib import Path

from ..step1_ingest.validators import ValidationResult
from ..step1_ingest.scanner import DatasetPart
from .yolo_converter import LABEL_MAP, ConversionResult
from .utils import to_yolo_line, write_label_file, find_actual_image

logger = logging.getLogger(__name__)


class CocoJsonConverter:
    """Converts one dataset part from COCO JSON format to YOLO .txt files."""

    def convert(
        self,
        validation: ValidationResult,
        part: DatasetPart,
        output_dir: Path,
    ) -> ConversionResult:
        logger.info("[coco_converter] Converting '%s'", part.name)

        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        stats = ConversionResult(part_name=part.name)

        with open(part.annotation_file, encoding="utf-8") as f:
            data = json.load(f)

        # Defined category ids
        defined_cat_ids = {cat["id"] for cat in data.get("categories", [])}

        # image_id → {file_name, width, height}
        id_to_image: dict[int, dict] = {
            img["id"]: img for img in data.get("images", [])
        }

        # Sets of invalid annotation_ids and image_ids from missing files
        invalid_ann_ids: set[int] = {
            b["annotation_id"]
            for b in validation.invalid_bboxes
            if "annotation_id" in b
        }
        missing_names: set[str] = set(validation.missing_images)

        # Group annotations by image_id
        annotations_by_image: dict[int, list[dict]] = {}
        for ann in data.get("annotations", []):
            iid = ann["image_id"]
            annotations_by_image.setdefault(iid, []).append(ann)

        for img_info in data.get("images", []):
            img_id   = img_info["id"]
            img_name = img_info["file_name"]
            img_w    = img_info.get("width",  0)
            img_h    = img_info.get("height", 0)

            # Find the real file (handles extension mismatch)
            actual_path = find_actual_image(img_name, part.images_dir, validation.orphan_files)
            if actual_path is None:
                if img_name in missing_names:
                    logger.debug("[coco_converter] Skipping missing file: %s", img_name)
                else:
                    logger.warning("[coco_converter] File not found: %s", img_name)
                stats.skipped_missing += 1
                continue

            if actual_path.name != img_name:
                logger.debug(
                    "[coco_converter] Extension mismatch: %s → %s", img_name, actual_path.name
                )
                stats.resolved_ext_mismatch += 1

            anns = annotations_by_image.get(img_id, [])
            yolo_lines: list[str] = []

            for ann in anns:
                ann_id = ann.get("id")
                cat_id = ann.get("category_id")
                bbox   = ann.get("bbox", [])

                # Skip invalid bbox
                if ann_id in invalid_ann_ids:
                    stats.skipped_invalid_bbox += 1
                    continue

                # Skip undefined / unknown category
                if cat_id not in defined_cat_ids:
                    stats.skipped_unknown_cat += 1
                    continue

                # Map category name → YOLO class_id
                cat_name = next(
                    (c["name"] for c in data["categories"] if c["id"] == cat_id), None
                )
                if cat_name is None or cat_name.lower() not in LABEL_MAP:
                    stats.skipped_unknown_cat += 1
                    continue

                class_id = LABEL_MAP[cat_name.lower()]

                # COCO bbox: [x_min, y_min, width, height]
                if len(bbox) != 4 or img_w <= 0 or img_h <= 0:
                    stats.skipped_invalid_bbox += 1
                    continue

                x, y, w, h = bbox
                cx = (x + w / 2) / img_w
                cy = (y + h / 2) / img_h
                bw = w / img_w
                bh = h / img_h

                # Clip to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))

                yolo_lines.append(to_yolo_line(class_id, cx, cy, bw, bh))
                stats.converted_annotations += 1

            # Copy image
            stem = Path(img_name).stem
            dest_img = images_out / actual_path.name
            if not dest_img.exists():
                shutil.copy2(actual_path, dest_img)

            # Write label file (by annotation stem — independent of extension)
            write_label_file(labels_out / f"{stem}.txt", yolo_lines)
            stats.converted_images += 1

        logger.info("[coco_converter] '%s' done: %s", part.name, stats)
        return stats
