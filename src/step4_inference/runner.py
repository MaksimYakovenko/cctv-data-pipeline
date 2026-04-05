"""
runner.py — Step 4 Inference
InferenceEngine  — runs YOLOv8 inference
BBoxDrawer       — draws bboxes using PIL
ReportWriter     — saves worst_10_report.json
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFile, ImageFont

# Allow PIL to read truncated files
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# Color palette per class_id (RGB)
CLASS_COLORS: list[tuple[int, int, int]] = [
    (255,  80,  80),   # 0 person  — red
    ( 80, 200,  80),   # 1 pet     — green
    ( 80, 130, 255),   # 2 vehicle — blue
]
DEFAULT_COLOR = (255, 200, 50)   # yellow for unknown classes


# ─── InferenceResult ──────────────────────────────────────────────────────────

@dataclass
class InferenceResult:
    """Inference result for a single image."""
    image_name:     str
    image_path:     Path
    max_confidence: float
    detections:     list[dict] = field(default_factory=list)
    reason:         str        = ""

    def to_dict(self, rank: int | None = None) -> dict:
        d: dict = {}
        if rank is not None:
            d["rank"] = rank
        d["image"]          = self.image_name
        d["max_confidence"] = round(self.max_confidence, 4)
        d["reason"]         = self.reason
        d["detections"]     = [
            {
                "class_id":   det["class_id"],
                "class_name": det["class_name"],
                "confidence": round(det["confidence"], 4),
                "bbox_xyxy":  [round(v, 1) for v in det["bbox_xyxy"]],
            }
            for det in self.detections
        ]
        return d

class InferenceEngine:
    """Loads YOLOv8 and runs inference on a folder of images."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model      = None

    def load_model(self) -> None:
        from ultralytics import YOLO
        logger.info("[inference] Loading model: %s", self.model_path)
        self.model = YOLO(self.model_path)
        logger.info("[inference] Model loaded")

    def run_on_dir(self, images_dir: Path) -> list[InferenceResult]:
        """Runs inference on every image in images_dir."""
        if self.model is None:
            self.load_model()

        import numpy as np

        images = sorted(
            p for p in images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        logger.info("[inference] Processing %d images from %s", len(images), images_dir)

        results: list[InferenceResult] = []
        for img_path in images:
            try:
                # Read via PIL → numpy (avoids format-related issues)
                img_np = np.array(Image.open(img_path).convert("RGB"))
                preds  = self.model(img_np, verbose=False)
                result = self._parse_prediction(img_path, preds[0])
            except Exception as exc:
                logger.warning("[inference] Error processing %s: %s", img_path.name, exc)
                result = InferenceResult(
                    image_name=img_path.name, image_path=img_path,
                    max_confidence=0.0, detections=[],
                    reason=f"error: {exc}",
                )
            results.append(result)

        logger.info("[inference] Inference complete")
        return results

    def _parse_prediction(self, img_path: Path, pred) -> InferenceResult:
        detections: list[dict] = []
        boxes = pred.boxes
        if boxes is not None and len(boxes) > 0:
            names = pred.names
            for i in range(len(boxes)):
                xyxy     = boxes.xyxy[i].tolist()
                conf     = float(boxes.conf[i])
                cls_id   = int(boxes.cls[i])
                cls_name = names.get(cls_id, str(cls_id))
                detections.append({
                    "bbox_xyxy":  xyxy,
                    "confidence": conf,
                    "class_id":   cls_id,
                    "class_name": cls_name,
                })

        if not detections:
            return InferenceResult(
                image_name=img_path.name, image_path=img_path,
                max_confidence=0.0, detections=[], reason="no detections",
            )

        max_conf = max(d["confidence"] for d in detections)
        return InferenceResult(
            image_name=img_path.name, image_path=img_path,
            max_confidence=max_conf, detections=detections,
            reason=f"max_conf={max_conf:.3f}",
        )

    @staticmethod
    def sort_worst(
        results: list[InferenceResult],
        top_n:   int = 10,
    ) -> list[InferenceResult]:
        return sorted(results, key=lambda r: r.max_confidence)[:top_n]


# ─── BBoxDrawer ───────────────────────────────────────────────────────────────

class BBoxDrawer:
    """Draws bounding boxes on images using PIL."""

    def draw_and_save(self, result: InferenceResult, output_dir: Path) -> Path:
        import shutil
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / result.image_name

        try:
            img  = Image.open(result.image_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            try:
                font = ImageFont.truetype("arial.ttf", size=18)
            except Exception:
                font = ImageFont.load_default()

            for det in result.detections:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                cls_id = det["class_id"]
                color  = CLASS_COLORS[cls_id] if cls_id < len(CLASS_COLORS) else DEFAULT_COLOR

                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

                label     = f"{det['class_name']} {det['confidence']:.0%}"
                text_pos  = (x1 + 2, max(0, y1 - 22))
                bbox_text = draw.textbbox(text_pos, label, font=font)
                draw.rectangle(bbox_text, fill=color)
                draw.text(text_pos, label, fill=(255, 255, 255), font=font)

            img.save(out_path)

        except Exception as exc:
            logger.warning(
                "[bbox_drawer] Failed to draw bbox for %s: %s — copying original",
                result.image_name, exc,
            )
            # If the file is corrupted — copy as-is (or create an empty placeholder)
            if result.image_path.exists():
                shutil.copy2(result.image_path, out_path)

        return out_path


# ─── ReportWriter ─────────────────────────────────────────────────────────────

class ReportWriter:
    """Saves worst_10_report.json."""

    def save(
        self,
        worst:           list[InferenceResult],
        output_dir:      Path,
        model_path:      str,
        inference_split: str,
        total_processed: int,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "generated_at":           datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "model":                  model_path,
            "inference_split":        inference_split,
            "total_images_processed": total_processed,
            "worst_10": [r.to_dict(rank=i + 1) for i, r in enumerate(worst)],
        }

        out_path = output_dir / "worst_10_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info("[inference] Report saved: %s", out_path)
        return out_path
