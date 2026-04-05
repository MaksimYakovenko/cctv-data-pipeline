"""
step3_split.dataset_splitter — Split Dataset
Splits the YOLO dataset into train/val/test splits.
Deterministic result via random.seed(42).
"""

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SEED        = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@dataclass
class SplitResult:
    """Split dataset statistics."""
    total:                int        = 0
    train_count:          int        = 0
    val_count:            int        = 0
    test_count:           int        = 0
    images_without_labels: int       = 0
    output_dir:           str        = ""
    files: dict[str, list[str]]      = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"SplitResult(total={self.total}, "
            f"train={self.train_count}, "
            f"val={self.val_count}, "
            f"test={self.test_count}, "
            f"no_label={self.images_without_labels})"
        )


# ─── DatasetSplitter ──────────────────────────────────────────────────────────

class DatasetSplitter:
    """
    Facade for Step 3 — Split Dataset.

    Example usage:
        splitter = DatasetSplitter("output/yolo_dataset", "output")
        result = splitter.run()
    """

    def __init__(
        self,
        yolo_dataset_dir: str | Path,
        output_path:      str | Path = "output",
    ) -> None:
        self.yolo_dir   = Path(yolo_dataset_dir)
        self.output_dir = Path(output_path) / "split_dataset"

    def _collect_pairs(self) -> list[tuple[Path, Path | None]]:
        """
        Collects (image_path, label_path | None) pairs.
        Images without a label are included as negative examples.
        """
        images_dir = self.yolo_dir / "images"
        labels_dir = self.yolo_dir / "labels"

        pairs: list[tuple[Path, Path | None]] = []

        for img in sorted(images_dir.iterdir()):
            if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label = labels_dir / f"{img.stem}.txt"
            if label.exists():
                pairs.append((img, label))
            else:
                logger.warning("[splitter] Image without label: %s", img.name)
                pairs.append((img, None))

        return pairs

    def _split_pairs(
        self, pairs: list[tuple[Path, Path | None]]
    ) -> dict[str, list[tuple[Path, Path | None]]]:
        """Deterministically shuffles and splits into train/val/test."""
        random.seed(SEED)
        shuffled = list(pairs)
        random.shuffle(shuffled)

        n       = len(shuffled)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        return {
            "train": shuffled[:n_train],
            "val":   shuffled[n_train : n_train + n_val],
            "test":  shuffled[n_train + n_val :],
        }

    def _copy_split(
        self,
        splits: dict[str, list[tuple[Path, Path | None]]],
    ) -> int:
        """Copies files into the respective subfolders. Returns the number of images without labels."""
        no_label_count = 0

        for split_name, pairs in splits.items():
            img_out = self.output_dir / split_name / "images"
            lbl_out = self.output_dir / split_name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for img_path, lbl_path in pairs:
                shutil.copy2(img_path, img_out / img_path.name)

                dest_label = lbl_out / f"{img_path.stem}.txt"
                if lbl_path is not None:
                    shutil.copy2(lbl_path, dest_label)
                else:
                    dest_label.touch()   # empty file = negative example
                    no_label_count += 1

        return no_label_count

    def _write_yaml(self) -> None:
        """Generates dataset.yaml for split_dataset."""
        src_yaml = self.yolo_dir / "dataset.yaml"
        nc    = 3
        names: list[str] = ["person", "pet", "vehicle"]

        if src_yaml.exists():
            text  = src_yaml.read_text(encoding="utf-8")
            lines = text.splitlines()
            parsed_names: list[str] = []
            in_names = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("nc:"):
                    try:
                        nc = int(stripped.split(":")[1].strip())
                    except ValueError:
                        pass
                if stripped == "names:":
                    in_names = True
                    continue
                if in_names and stripped.startswith("- "):
                    parsed_names.append(stripped[2:].strip())
                elif in_names and stripped and not stripped.startswith("-"):
                    in_names = False
            if parsed_names:
                names = parsed_names

        names_str = "\n".join(f"  - {n}" for n in names)
        content = (
            f"path: {self.output_dir.resolve()}\n"
            f"train: train/images\n"
            f"val: val/images\n"
            f"test: test/images\n"
            f"\n"
            f"nc: {nc}\n"
            f"names:\n{names_str}\n"
        )
        yaml_path = self.output_dir / "dataset.yaml"
        yaml_path.write_text(content, encoding="utf-8")
        logger.info("[splitter] dataset.yaml saved: %s", yaml_path)

    def _write_metadata(
        self,
        splits: dict[str, list[tuple[Path, Path | None]]],
        result: SplitResult,
    ) -> None:
        """Saves metadata.json for reproducibility."""
        metadata = {
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "source":       str(self.yolo_dir),
            "seed":         SEED,
            "ratios": {
                "train": TRAIN_RATIO,
                "val":   VAL_RATIO,
                "test":  round(1 - TRAIN_RATIO - VAL_RATIO, 2),
            },
            "counts": {
                "total": result.total,
                "train": result.train_count,
                "val":   result.val_count,
                "test":  result.test_count,
                "images_without_labels": result.images_without_labels,
            },
            "files": {
                split: [img.name for img, _ in pairs]
                for split, pairs in splits.items()
            },
        }
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        logger.info("[splitter] metadata.json saved: %s", meta_path)

    # ── public interface ───────────────────────────────────────────────────────

    def run(self) -> SplitResult:
        """Runs the split and returns a SplitResult."""
        logger.info("[splitter] Starting split: %s", self.yolo_dir)

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        pairs  = self._collect_pairs()
        splits = self._split_pairs(pairs)

        no_label_count = self._copy_split(splits)

        result = SplitResult(
            total                 = len(pairs),
            train_count           = len(splits["train"]),
            val_count             = len(splits["val"]),
            test_count            = len(splits["test"]),
            images_without_labels = no_label_count,
            output_dir            = str(self.output_dir),
            files={
                split: [img.name for img, _ in ps]
                for split, ps in splits.items()
            },
        )

        self._write_yaml()
        self._write_metadata(splits, result)

        logger.info(
            "[splitter] ══ Step 3 Summary ══ "
            "total=%d | train=%d | val=%d | test=%d | no_labels=%d",
            result.total, result.train_count,
            result.val_count, result.test_count,
            result.images_without_labels,
        )
        return result
