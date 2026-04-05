import json
import logging
from datetime import datetime
from pathlib import Path

from .validators import ValidationResult

logger = logging.getLogger(__name__)

_MAX_LIST_ITEMS = 20


def _find_extension_mismatches(result: ValidationResult) -> list[dict]:
    mismatches = []
    orphan_stems = {Path(f).stem: f for f in result.orphan_files}

    for missing in result.missing_images:
        stem = Path(missing).stem
        if stem in orphan_stems:
            mismatches.append({
                "in_annotation": missing,
                "on_disk": orphan_stems[stem],
            })
    return mismatches


def _build_summary(results: list[ValidationResult]) -> dict:
    return {
        "total_parts": len(results),
        "total_images_in_annotations": sum(
            r.total_images_in_annotation for r in results),
        "total_images_on_disk": sum(r.total_images_on_disk for r in results),
        "total_annotations": sum(r.total_annotations for r in results),
        "total_missing_images": sum(len(r.missing_images) for r in results),
        "total_orphan_files": sum(len(r.orphan_files) for r in results),
        "total_invalid_bboxes": sum(len(r.invalid_bboxes) for r in results),
        "total_unknown_category_ids": sum(
            len(r.unknown_category_ids) for r in results),
    }


def _serialize_part(result: ValidationResult) -> dict:
    return {
        "part_name": result.part_name,
        "format": result.format.value,
        "is_valid": result.is_valid,
        "total_images_in_annotation": result.total_images_in_annotation,
        "total_images_on_disk": result.total_images_on_disk,
        "total_annotations": result.total_annotations,
        "missing_images": result.missing_images,
        "orphan_files": result.orphan_files,
        "extension_mismatches": _find_extension_mismatches(result),
        "invalid_bboxes": result.invalid_bboxes,
        "unknown_category_ids": result.unknown_category_ids,
    }


def _fmt_list(items: list, label: str, lines: list[str]) -> None:
    lines.append(f"  {label} ({len(items)}):")
    if not items:
        lines.append("    (none)")
        return
    for item in items[:_MAX_LIST_ITEMS]:
        if isinstance(item, dict):
            lines.append(f"    - {item}")
        else:
            lines.append(f"    - {item}")
    if len(items) > _MAX_LIST_ITEMS:
        lines.append(f"    ... and {len(items) - _MAX_LIST_ITEMS} more")


class ValidationReporter:
    def _write_json(
            self,
            results: list[ValidationResult],
            summary: dict,
            out_path: Path,
            ts: str,
    ) -> None:
        data = {
            "generated_at": ts,
            "summary": summary,
            "parts": [_serialize_part(r) for r in results],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info("[reporter] JSON звіт збережено: %s", out_path)

    def _write_txt(
            self,
            results: list[ValidationResult],
            summary: dict,
            out_path: Path,
            ts: str,
    ) -> None:
        W = 54
        lines: list[str] = []

        lines.append("=" * W)
        lines.append("  VALIDATION REPORT — Step 1 Ingest")
        lines.append(f"  Generated: {ts}")
        lines.append("=" * W)
        lines.append("")

        lines.append("SUMMARY")
        lines.append("-" * W)
        lines.append(f"  Total parts:               {summary['total_parts']}")
        lines.append(
            f"  Total images (annotation): {summary['total_images_in_annotations']}")
        lines.append(
            f"  Total images (disk):       {summary['total_images_on_disk']}")
        lines.append(
            f"  Total annotations:         {summary['total_annotations']}")
        lines.append(
            f"  Missing images:            {summary['total_missing_images']}")
        lines.append(
            f"  Orphan files:              {summary['total_orphan_files']}")
        lines.append(
            f"  Invalid bboxes:            {summary['total_invalid_bboxes']}")
        lines.append(
            f"  Unknown category IDs:      {summary['total_unknown_category_ids']}")
        lines.append("")

        # ── кожна частина ─────────────────────────────────────
        for r in results:
            status = "VALID" if r.is_valid else "INVALID"
            lines.append("─" * W)
            lines.append(f"  PART: {r.part_name}  [{r.format.value}]  {status}")
            lines.append("─" * W)
            lines.append(
                f"  Images in annotation:  {r.total_images_in_annotation}")
            lines.append(f"  Images on disk:        {r.total_images_on_disk}")
            lines.append(f"  Total annotations:     {r.total_annotations}")
            lines.append("")

            mismatches = _find_extension_mismatches(r)

            _fmt_list(r.missing_images, "Missing images", lines)
            lines.append("")
            _fmt_list(r.orphan_files,
                      "Orphan files (on disk, not in annotation)", lines)
            lines.append("")

            if mismatches:
                lines.append(f"  Extension mismatches ({len(mismatches)}):")
                for m in mismatches:
                    lines.append(f"    - annotation: {m['in_annotation']}")
                    lines.append(f"      on disk:    {m['on_disk']}")
                lines.append("")

            if r.unknown_category_ids:
                lines.append(
                    f"  Unknown category_ids: {r.unknown_category_ids}")
                lines.append("")

            _fmt_list(r.invalid_bboxes, "Invalid bboxes", lines)
            lines.append("")

        lines.append("=" * W)
        lines.append("  END OF REPORT")
        lines.append("=" * W)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("[reporter] TXT звіт збережено: %s", out_path)

    def generate(
            self,
            results: list[ValidationResult],
            output_dir: str | Path,
    ) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        summary = _build_summary(results)

        self._write_json(results, summary, out / "report.json", ts)
        self._write_txt(results, summary, out / "report.txt", ts)

        logger.info("[reporter] Звіт згенеровано у: %s", out)
