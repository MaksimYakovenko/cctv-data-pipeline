"""
xml_converter.py — Step 2 Convert
Converts CVAT XML annotations to YOLO format.
"""

import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from ..step1_ingest.validators import ValidationResult
from ..step1_ingest.scanner import DatasetPart
from .yolo_converter import LABEL_MAP, ConversionResult
from .utils import to_yolo_line, write_label_file, find_actual_image

logger = logging.getLogger(__name__)


class CvatXmlConverter:
    """Converts one dataset part from CVAT XML format to YOLO .txt files."""

    def convert(
        self,
        validation: ValidationResult,
        part: DatasetPart,
        output_dir: Path,
    ) -> ConversionResult:
        logger.info("[cvat_converter] Converting '%s'", part.name)

        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        stats = ConversionResult(part_name=part.name)

        tree = ET.parse(part.annotation_file)
        root = tree.getroot()

        # Build a set of invalid bbox keys for fast lookup
        # Key: (image_name, label, xtl, ytl, xbr, ybr)
        invalid_bbox_keys: set[tuple] = set()
        for b in validation.invalid_bboxes:
            if "coords" in b:
                c = b["coords"]
                invalid_bbox_keys.add((
                    b.get("image_name", ""),
                    b.get("label", ""),
                    c.get("xtl"), c.get("ytl"), c.get("xbr"), c.get("ybr"),
                ))

        for img_elem in root.findall("image"):
            img_name = img_elem.get("name", "")
            img_w    = int(img_elem.get("width",  0))
            img_h    = int(img_elem.get("height", 0))

            # Find the real file on disk
            actual_path = find_actual_image(img_name, part.images_dir, validation.orphan_files)
            if actual_path is None:
                logger.debug("[cvat_converter] Skipping missing file: %s", img_name)
                stats.skipped_missing += 1
                continue

            if actual_path.name != img_name:
                logger.debug(
                    "[cvat_converter] Extension mismatch: %s → %s", img_name, actual_path.name
                )
                stats.resolved_ext_mismatch += 1

            yolo_lines: list[str] = []

            for box in img_elem.findall("box"):
                label = box.get("label", "").lower()
                xtl   = float(box.get("xtl", 0))
                ytl   = float(box.get("ytl", 0))
                xbr   = float(box.get("xbr", 0))
                ybr   = float(box.get("ybr", 0))

                # Skip invalid bbox
                key = (img_name, box.get("label", ""), xtl, ytl, xbr, ybr)
                if key in invalid_bbox_keys:
                    stats.skipped_invalid_bbox += 1
                    continue

                # Map label → YOLO class_id
                if label not in LABEL_MAP:
                    logger.warning(
                        "[cvat_converter] Unknown label '%s' in %s", label, img_name
                    )
                    stats.skipped_unknown_cat += 1
                    continue

                class_id = LABEL_MAP[label]

                if img_w <= 0 or img_h <= 0:
                    stats.skipped_invalid_bbox += 1
                    continue

                # CVAT bbox: xtl,ytl,xbr,ybr → YOLO cx,cy,w,h normalised
                bw = xbr - xtl
                bh = ybr - ytl
                cx = (xtl + bw / 2) / img_w
                cy = (ytl + bh / 2) / img_h
                bw = bw / img_w
                bh = bh / img_h

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

            # Write label file
            write_label_file(labels_out / f"{stem}.txt", yolo_lines)
            stats.converted_images += 1

        logger.info("[cvat_converter] '%s' done: %s", part.name, stats)
        return stats
