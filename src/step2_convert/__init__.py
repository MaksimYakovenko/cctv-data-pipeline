"""
step2_convert — converts annotations to YOLO format.
"""

import logging
from pathlib import Path

from ..step1_ingest.scanner import AnnotationFormat, DatasetScanner
from ..step1_ingest.validators import ValidationResult
from .yolo_converter import CLASS_NAMES, ConversionResult
from .json_converter import CocoJsonConverter
from .xml_converter import CvatXmlConverter

logger = logging.getLogger(__name__)


class ConversionPipeline:
    """
    Facade for Step 2 — Convert to YOLO.
    Converts all dataset parts to YOLO format and generates dataset.yaml.

    Example usage:
        results = ConversionPipeline(validated, raw_data_path, output_path).run()
    """

    def __init__(
        self,
        validation_results: list[ValidationResult],
        raw_data_path:      str = "data/raw_dataset",
        output_path:        str = "output",
    ) -> None:
        self.validation_results = validation_results
        self.raw_data_path      = raw_data_path
        self.output_path        = Path(output_path)

    def run(self) -> list[ConversionResult]:
        """Runs the conversion and returns a list of ConversionResult."""
        yolo_dir = self.output_path / "yolo_dataset"
        yolo_dir.mkdir(parents=True, exist_ok=True)

        # Get DatasetPart for each part
        parts = DatasetScanner(self.raw_data_path).scan()
        parts_by_name = {p.name: p for p in parts}

        coco_conv = CocoJsonConverter()
        cvat_conv = CvatXmlConverter()

        conversion_results: list[ConversionResult] = []

        for val_result in self.validation_results:
            part = parts_by_name.get(val_result.part_name)
            if part is None:
                logger.warning(
                    "[conversion] Part '%s' not found during scanning — skipping",
                    val_result.part_name,
                )
                continue

            if part.format == AnnotationFormat.COCO_JSON:
                result = coco_conv.convert(val_result, part, yolo_dir)
            elif part.format == AnnotationFormat.CVAT_XML:
                result = cvat_conv.convert(val_result, part, yolo_dir)
            else:
                logger.warning(
                    "[conversion] Unsupported format '%s' — skipping",
                    part.format,
                )
                continue

            conversion_results.append(result)

        self._write_dataset_yaml(yolo_dir)
        self._log_summary(conversion_results)

        return conversion_results

    def _write_dataset_yaml(self, yolo_dir: Path) -> None:
        """Generates dataset.yaml for YOLOv8."""
        yaml_path = yolo_dir / "dataset.yaml"
        names_str = "\n".join(f"  - {name}" for name in CLASS_NAMES)
        content = (
            f"path: {yolo_dir.resolve()}\n"
            f"train: images\n"
            f"val: images\n"
            f"test: images\n"
            f"\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names:\n{names_str}\n"
        )
        yaml_path.write_text(content, encoding="utf-8")
        logger.info("[conversion] dataset.yaml saved: %s", yaml_path)

    def _log_summary(self, results: list[ConversionResult]) -> None:
        total_images = sum(r.converted_images      for r in results)
        total_anns   = sum(r.converted_annotations for r in results)
        total_skip   = sum(r.skipped_missing        for r in results)
        total_inv    = sum(r.skipped_invalid_bbox   for r in results)
        total_unk    = sum(r.skipped_unknown_cat    for r in results)
        total_fixed  = sum(r.resolved_ext_mismatch  for r in results)

        logger.info(
            "[conversion] ══ Step 2 Summary ══ "
            "images=%d | annotations=%d | "
            "skipped(missing)=%d | skipped(invalid_bbox)=%d | "
            "skipped(unknown_cat)=%d | fixed(ext_mismatch)=%d",
            total_images, total_anns,
            total_skip, total_inv, total_unk, total_fixed,
        )


__all__ = ["ConversionPipeline", "ConversionResult"]
