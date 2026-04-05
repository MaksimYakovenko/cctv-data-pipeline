import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
sys.path.insert(0, "src")

from step1_ingest.scanner import DatasetScanner
from step1_ingest.validators import DatasetValidator
from step1_ingest.report import ValidationReporter

parts = DatasetScanner("data/raw_dataset").scan()
validator = DatasetValidator()
results = []

for part in parts:
    result = validator.validate(part)
    results.append(result)
    print()
    print(result)

print()
ValidationReporter().generate(results, "output/validation_report_results_step_1")
