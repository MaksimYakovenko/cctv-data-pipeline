import logging

from src.step1_ingest import DataIngestor
from src.step2_convert import ConversionPipeline
from src.step3_split import DatasetSplitter
from src.step4_inference import InferenceRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(message)s",
)


def run_pipeline(
    raw_data_path: str = "data/raw_dataset",
    output_path:   str = "output",
    model_path:    str = "yolov8n.pt",
) -> None:
    # ── Step 1: Ingest ────────────────────────────────────
    ingestor  = DataIngestor(raw_data_path, output_path)
    validated = ingestor.run()   # → list[ValidationResult]
                                 # → output/validation_report_results_step_1/

    # ── Step 2: Convert to YOLO ───────────────────────────
    converter = ConversionPipeline(validated, raw_data_path, output_path)
    yolo_data = converter.run()  # → list[ConversionResult]
                                 # → output/yolo_dataset/images/
                                 # → output/yolo_dataset/labels/
                                 # → output/yolo_dataset/dataset.yaml

    # ── Step 3: Split Dataset ─────────────────────────────
    splitter = DatasetSplitter(f"{output_path}/yolo_dataset", output_path)
    dataset  = splitter.run()    # → SplitResult
                                 # → output/split_dataset/train|val|test/
                                 # → output/split_dataset/dataset.yaml
                                 # → output/split_dataset/metadata.json

    # ── Step 4: Inference ─────────────────────────────────
    runner  = InferenceRunner(
        split_dataset_dir = f"{output_path}/split_dataset",
        output_path       = output_path,
        model_path        = model_path,
        split             = "val",
        top_n             = 10,
    )
    runner.run()                 # → output/inference_results/*.jpg (10 worst)
                                 # → output/inference_results/worst_10_report.json


if __name__ == "__main__":
    run_pipeline()
