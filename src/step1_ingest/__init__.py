from .scanner import AnnotationFormat, DatasetPart, DatasetScanner
from .validators import DatasetValidator, ValidationResult
from .report import ValidationReporter


class DataIngestor:
    """
    Facade for Step 1 — Ingest.
    Scans the dataset, validates each part, generates a Validation Report.
    """

    def __init__(self, raw_data_path: str, output_path: str = "output") -> None:
        self.raw_data_path = raw_data_path
        self.output_path   = output_path

    def run(self) -> list[ValidationResult]:
        """Runs Step 1 and returns a list of ValidationResult."""
        # 1. Scan
        parts = DatasetScanner(self.raw_data_path).scan()

        # 2. Validate
        validator = DatasetValidator()
        results   = [validator.validate(part) for part in parts]

        # 3. Report
        report_dir = f"{self.output_path}/validation_report_results_step_1"
        ValidationReporter().generate(results, report_dir)

        return results


__all__ = [
    "AnnotationFormat",
    "DatasetPart",
    "DatasetScanner",
    "DatasetValidator",
    "ValidationResult",
    "ValidationReporter",
    "DataIngestor",
]
