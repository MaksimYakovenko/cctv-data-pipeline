# CCTV Data Pipeline

## Overview

A four-step pipeline for processing CCTV image datasets — from raw annotations to YOLOv8 inference.

---

## Requirements

- Python 3.10+
- Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):
- `ultralytics>=8.0.0` — YOLOv8
- `Pillow>=10.0.0` — image processing

---

## Usage

```bash
python pipeline.py
```

Default parameters:
| Parameter       | Default value          |
|-----------------|------------------------|
| `raw_data_path` | `data/raw_dataset`     |
| `output_path`   | `output`               |
| `model_path`    | `yolov8n.pt`           |

---

## Pipeline Algorithm

### Step 1 — Ingestion & Validation (`step1_ingest`)

1. **Scan** the `data/raw_dataset/` directory.  
   Dataset parts (`ds_part1`, `ds_part2`, …) are discovered.  
   For each part the following is detected:
   - annotation format: **COCO JSON** (`.json`) or **CVAT XML** (`.xml`);
   - images directory (`data/`).

2. **Validate** each part:
   - cross-check images against annotations;
   - collect warnings for missing or mismatched files.

3. **Report** is saved to:
   ```
   output/validation_report_results_step_1/
       report.json
       report.txt
   ```

---

### Step 2 — Convert to YOLO Format (`step2_convert`)

1. The appropriate converter is run for each valid dataset part:
   - `json_converter.py` — COCO JSON → YOLO TXT;
   - `xml_converter.py` — CVAT XML → YOLO TXT.

2. Images and labels are copied/converted into a unified dataset:
   ```
   output/yolo_dataset/
       images/          ← all images
       labels/          ← YOLO-format annotation files (.txt)
       dataset.yaml     ← class configuration
   ```

---

### Step 3 — Dataset Split (`step3_split`)

1. The dataset from `output/yolo_dataset/` is split into three subsets:
   - **train** — training set;
   - **val** — validation set;
   - **test** — test set.

2. Results are saved to:
   ```
   output/split_dataset/
       train/images/  train/labels/
       val/images/    val/labels/
       test/images/   test/labels/
       dataset.yaml   ← subset paths and class list
       metadata.json  ← split statistics
   ```

---

### Step 4 — YOLOv8 Inference (`step4_inference`)

1. Model `yolov8n.pt` is loaded.

2. Inference is run on the **val** split from `output/split_dataset/val/images/`.

3. For each image:
   - object detection is performed (classes: `person`, `pet`, `vehicle`);
   - maximum confidence score (`max_confidence`) is recorded.

4. The **10 images with the lowest confidence** are selected (`top_n=10`).

5. Bounding boxes are drawn on images using PIL:
   - 🔴 red — `person`
   - 🟢 green — `pet`
   - 🔵 blue — `vehicle`

6. Results are saved to:
   ```
   output/inference_results/
       *.jpg / *.png        ← 10 worst-confidence images with bboxes
       worst_10_report.json ← detailed per-image report
   ```

---

## Output Structure

```
output/
├── validation_report_results_step_1/   # Step 1: validation report
│   ├── report.json
│   └── report.txt
├── yolo_dataset/                       # Step 2: YOLO-format dataset
│   ├── images/
│   ├── labels/
│   └── dataset.yaml
├── split_dataset/                      # Step 3: train/val/test split
│   ├── train/ val/ test/
│   ├── dataset.yaml
│   └── metadata.json
└── inference_results/                  # Step 4: inference results
    ├── *.jpg / *.png
    └── worst_10_report.json
```
