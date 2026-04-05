"""
step4_inference — Step 4: Inference
Runs YOLOv8n on the val split, finds the 10 worst detections,
draws bounding boxes and saves a report.
"""

import logging
from pathlib import Path

from .runner import BBoxDrawer, InferenceEngine, InferenceResult, ReportWriter

logger = logging.getLogger(__name__)


class InferenceRunner:
    """
    Facade for Step 4 — Inference.

    Example usage:
        runner = InferenceRunner("output/split_dataset", "output", "yolov8n.pt")
        results = runner.run()
    """

    def __init__(
        self,
        split_dataset_dir: str | Path,
        output_path:       str | Path = "output",
        model_path:        str        = "yolov8n.pt",
        split:             str        = "val",
        top_n:             int        = 10,
    ) -> None:
        self.split_dir  = Path(split_dataset_dir)
        self.output_dir = Path(output_path) / "inference_results"
        self.model_path = model_path
        self.split      = split
        self.top_n      = top_n

    def run(self) -> list[InferenceResult]:
        """Runs inference and returns a list of the 10 worst InferenceResults."""
        images_dir = self.split_dir / self.split / "images"
        if not images_dir.exists():
            logger.error("[inference] Folder not found: %s", images_dir)
            return []

        # 1. Inference
        engine  = InferenceEngine(self.model_path)
        all_results = engine.run_on_dir(images_dir)

        # 2. Select the 10 worst
        worst = InferenceEngine.sort_worst(all_results, self.top_n)

        logger.info(
            "[inference] Top-%d worst (max_conf): %s",
            self.top_n,
            [f"{r.image_name}={r.max_confidence:.3f}" for r in worst],
        )

        # 3. Draw bboxes and save images
        drawer = BBoxDrawer()
        for result in worst:
            saved = drawer.draw_and_save(result, self.output_dir)
            logger.info("[inference] Saved: %s", saved)

        # 4. Save JSON report
        ReportWriter().save(
            worst           = worst,
            output_dir      = self.output_dir,
            model_path      = self.model_path,
            inference_split = self.split,
            total_processed = len(all_results),
        )

        logger.info(
            "[inference] ══ Step 4 Summary ══ "
            "processed=%d | saved worst=%d | output=%s",
            len(all_results), len(worst), self.output_dir,
        )
        return worst


__all__ = ["InferenceRunner", "InferenceResult"]
