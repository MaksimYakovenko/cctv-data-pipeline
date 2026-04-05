"""
yolo_converter.py — Step 2 Convert
LABEL_MAP, CLASS_NAMES and ConversionResult dataclass.
"""

from dataclasses import dataclass, field

LABEL_MAP: dict[str, int] = {
    "person": 0,
    "human": 0,
    "pet": 1,
    "vehicle": 2,
    "car": 2,
}

CLASS_NAMES: list[str] = ["person", "pet", "vehicle"]


# ─── ConversionResult ─────────────────────────────────────────────────────────

@dataclass
class ConversionResult:
    """Conversion statistics of one part of the dataset."""
    part_name: str
    converted_images: int = 0
    converted_annotations: int = 0
    skipped_missing: int = 0
    skipped_invalid_bbox: int = 0
    skipped_unknown_cat: int = 0
    skipped_ext_mismatch: int = 0
    resolved_ext_mismatch: int = 0
    errors: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"ConversionResult(part={self.part_name!r}, "
            f"images={self.converted_images}, "
            f"annotations={self.converted_annotations}, "
            f"skipped_missing={self.skipped_missing}, "
            f"skipped_invalid={self.skipped_invalid_bbox}, "
            f"skipped_unknown_cat={self.skipped_unknown_cat}, "
            f"resolved_ext={self.resolved_ext_mismatch})"
        )

