import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

from .scanner import AnnotationFormat, DatasetPart

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    part_name: str
    format: AnnotationFormat
    total_images_in_annotation: int
    total_images_on_disk: int
    missing_images: list[str] = field(default_factory=list)
    orphan_files: list[str] = field(default_factory=list)
    total_annotations: int = 0
    invalid_bboxes: list[dict] = field(default_factory=list)
    unknown_category_ids: list[int] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.missing_images) == 0 and len(self.invalid_bboxes) == 0

    def __repr__(self) -> str:
        return (
            f"ValidationResult(part={self.part_name!r}, valid={self.is_valid}, "
            f"missing={len(self.missing_images)}, "
            f"orphans={len(self.orphan_files)}, "
            f"invalid_bboxes={len(self.invalid_bboxes)}, "
            f"unknown_cats={len(self.unknown_category_ids)})"
        )


def _log_result(r: ValidationResult) -> None:
    if r.missing_images:
        logger.warning(
            "[validator] '%s': %d відсутніх файлів: %s",
            r.part_name, len(r.missing_images), r.missing_images,
        )
    if r.orphan_files:
        logger.warning(
            "[validator] '%s': %d orphan-файлів (є на диску, немає в анотації): %s",
            r.part_name, len(r.orphan_files), r.orphan_files,
        )
    if r.invalid_bboxes:
        logger.warning(
            "[validator] '%s': %d невалідних bbox",
            r.part_name, len(r.invalid_bboxes),
        )
        for bbox_issue in r.invalid_bboxes:
            logger.warning("  → %s", bbox_issue)
    if r.unknown_category_ids:
        logger.warning(
            "[validator] '%s': невизначені category_id=%s",
            r.part_name, r.unknown_category_ids,
        )
    status = "✓ валідна" if r.is_valid else "✗ є проблеми"
    logger.info("[validator] '%s': %s", r.part_name, status)


class DatasetValidator:

    def _validate_coco_json(self, part: DatasetPart) -> ValidationResult:
        logger.info("[validator] COCO JSON: читаємо %s", part.annotation_file)

        with open(part.annotation_file, encoding="utf-8") as f:
            data = json.load(f)

        # --- зображення ---
        annotation_images: dict[int, str] = {
            img["id"]: img["file_name"] for img in data.get("images", [])
        }
        annotation_names = set(annotation_images.values())
        disk_names = set(part.image_files)

        missing_images = sorted(annotation_names - disk_names)
        orphan_files = sorted(disk_names - annotation_names)

        defined_cat_ids = {cat["id"] for cat in data.get("categories", [])}
        annotations = data.get("annotations", [])

        unknown_cat_ids_set: set[int] = set()
        invalid_bboxes: list[dict] = []

        for ann in annotations:
            cat_id = ann.get("category_id")

            if cat_id not in defined_cat_ids:
                unknown_cat_ids_set.add(cat_id)

            bbox = ann.get("bbox", [])
            if len(bbox) == 4:
                x, y, w, h = bbox
                if w <= 0 or h <= 0:
                    invalid_bboxes.append({
                        "annotation_id": ann.get("id"),
                        "image_id": ann.get("image_id"),
                        "image_name": annotation_images.get(ann.get("image_id"),
                                                            "unknown"),
                        "bbox": bbox,
                        "reason": f"w={w}, h={h} — нульова або від'ємна розмірність",
                    })
            else:
                invalid_bboxes.append({
                    "annotation_id": ann.get("id"),
                    "image_id": ann.get("image_id"),
                    "image_name": annotation_images.get(ann.get("image_id"),
                                                        "unknown"),
                    "bbox": bbox,
                    "reason": f"некоректний формат bbox: очікується 4 елементи, отримано {len(bbox)}",
                })

        result = ValidationResult(
            part_name=part.name,
            format=part.format,
            total_images_in_annotation=len(annotation_images),
            total_images_on_disk=len(disk_names),
            missing_images=missing_images,
            orphan_files=orphan_files,
            total_annotations=len(annotations),
            invalid_bboxes=invalid_bboxes,
            unknown_category_ids=sorted(unknown_cat_ids_set),
        )

        _log_result(result)
        return result

    # ── CVAT XML ───────────────────────────────────────────────────────────────

    def _validate_cvat_xml(self, part: DatasetPart) -> ValidationResult:
        tree = ET.parse(part.annotation_file)
        root = tree.getroot()

        xml_images = root.findall("image")

        annotation_names = {img.get("name") for img in xml_images}
        disk_names = set(part.image_files)

        missing_images = sorted(annotation_names - disk_names)
        orphan_files = sorted(disk_names - annotation_names)

        total_annotations = 0
        invalid_bboxes: list[dict] = []

        for img_elem in xml_images:
            img_name = img_elem.get("name", "unknown")
            img_w = int(img_elem.get("width", 0))
            img_h = int(img_elem.get("height", 0))

            for box in img_elem.findall("box"):
                total_annotations += 1
                label = box.get("label", "")
                xtl = float(box.get("xtl", 0))
                ytl = float(box.get("ytl", 0))
                xbr = float(box.get("xbr", 0))
                ybr = float(box.get("ybr", 0))

                reasons: list[str] = []

                # нульова або від'ємна площа
                if xbr <= xtl or ybr <= ytl:
                    reasons.append(
                        f"нульова площа: xbr({xbr}) <= xtl({xtl}) або ybr({ybr}) <= ytl({ytl})"
                    )

                # out-of-bounds
                if xtl < 0:
                    reasons.append(f"xtl={xtl} < 0")
                if ytl < 0:
                    reasons.append(f"ytl={ytl} < 0")
                if img_w > 0 and xbr > img_w:
                    reasons.append(f"xbr={xbr} > width={img_w}")
                if img_h > 0 and ybr > img_h:
                    reasons.append(f"ybr={ybr} > height={img_h}")

                if reasons:
                    invalid_bboxes.append({
                        "image_name": img_name,
                        "label": label,
                        "coords": {"xtl": xtl, "ytl": ytl, "xbr": xbr,
                                   "ybr": ybr},
                        "reason": "; ".join(reasons),
                    })

        result = ValidationResult(
            part_name=part.name,
            format=part.format,
            total_images_in_annotation=len(annotation_names),
            total_images_on_disk=len(disk_names),
            missing_images=missing_images,
            orphan_files=orphan_files,
            total_annotations=total_annotations,
            invalid_bboxes=invalid_bboxes,
            unknown_category_ids=[],
        )

        _log_result(result)
        return result

    def validate(self, part: DatasetPart) -> ValidationResult:
        logger.info(
            "[validator] validation started: '%s' (%s)",
            part.name, part.format.value,
        )
        if part.format == AnnotationFormat.COCO_JSON:
            return self._validate_coco_json(part)
        elif part.format == AnnotationFormat.CVAT_XML:
            return self._validate_cvat_xml(part)
        else:
            raise ValueError(f"Not supported file format: {part.format}")
