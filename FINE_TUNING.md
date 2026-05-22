# Fine-Tuning the Car Damage Detection Model

This guide explains how to improve the YOLOv8 damage detection model (`best.pt`) by training it on new labeled images using the `finetune.py` script.

---

## Overview

The detection model (`best.pt`) is a YOLOv8 model trained to recognize **11 car damage classes**:

| ID | Class Name |
|----|------------|
| 0  | damage-front-windscreen |
| 1  | damaged-door |
| 2  | damaged-fender |
| 3  | damaged-front-bumper |
| 4  | damaged-head-light |
| 5  | damaged-hood |
| 6  | damaged-rear-bumper |
| 7  | damaged-rear-window |
| 8  | damaged-side-window |
| 9  | damaged-tail-light |
| 10 | quaterpanel-dent |

**Why fine-tune?** The base model was trained on a general dataset. If it misses damage types that are common in your specific vehicle fleet, or produces too many false positives for particular damage locations, fine-tuning on your own labeled images will improve its accuracy for your use case.

**When to fine-tune:**
- The model frequently misses a specific damage class (e.g., consistently ignores `damaged-fender` on certain car models).
- The model produces too many false positives for a specific class on your fleet's typical camera angle.
- You have collected 50+ new labeled examples that are not represented in the original training data.

The fine-tuning process is **additive** — it starts from the existing `best.pt` weights and specializes them further. You do not need to train from scratch.

---

## Prerequisites

- Python environment with ultralytics installed:
  ```bash
  pip install ultralytics
  ```
- **GPU strongly recommended.** CUDA-enabled training is 10–50x faster than CPU. CPU training is possible but can take hours per epoch for a typical dataset.
- Minimum recommended dataset size: **50+ images per class** you want to improve, with an 80/20 train/val split.
- `best.pt` present in the project root (or path provided via `--base-model`).

---

## Step 1: Collect and Label Training Images

Images must be labeled in **YOLO format** before training.

### Recommended labeling tools
- [LabelImg](https://github.com/HumanSignal/labelImg) — free, desktop, outputs YOLO `.txt` files directly.
- [Roboflow](https://roboflow.com) — web-based, supports team annotation, exports in YOLO format.

### Labeling rules

Each label file is a plain `.txt` file with one detection per line:
```
<class_id> <cx> <cy> <width> <height>
```
All values are normalized to [0, 1] relative to the image dimensions. Example:
```
3 0.512 0.437 0.210 0.155
```

**IMPORTANT: Use only the 11 existing class IDs (0–10).** The IDs and class names are:

| ID | Class Name |
|----|------------|
| 0  | damage-front-windscreen |
| 1  | damaged-door |
| 2  | damaged-fender |
| 3  | damaged-front-bumper |
| 4  | damaged-head-light |
| 5  | damaged-hood |
| 6  | damaged-rear-bumper |
| 7  | damaged-rear-window |
| 8  | damaged-side-window |
| 9  | damaged-tail-light |
| 10 | quaterpanel-dent |

> **Warning:** Adding new class names or changing the `nc` value will break compatibility with the existing model structure and require retraining from scratch. Keep the class vocabulary at exactly these 11 names.

---

## Step 2: Prepare Dataset Directory Structure

Organize your files in the following layout before running the script:

```
my_dataset/
  images/
    train/   (your training images: .jpg or .png)
    val/     (your validation images)
  labels/
    train/   (YOLO .txt label files, same stem as corresponding image)
    val/
  data.yaml
```

Each image in `images/train/` must have a corresponding label file in `labels/train/` with the same filename stem. For example:
- `images/train/car_001.jpg` → `labels/train/car_001.txt`

### data.yaml template

Create `my_dataset/data.yaml` with the following content (replace the `path` value):

```yaml
path: /absolute/path/to/my_dataset
train: images/train
val: images/val
nc: 11
names:
  - damage-front-windscreen
  - damaged-door
  - damaged-fender
  - damaged-front-bumper
  - damaged-head-light
  - damaged-hood
  - damaged-rear-bumper
  - damaged-rear-window
  - damaged-side-window
  - damaged-tail-light
  - quaterpanel-dent
```

> **Note:** `path` must be an **absolute** path. The `nc` value and `names` list must match exactly — do not add, remove, or reorder entries.

---

## Step 3: Run the Fine-Tuning Script

### Basic usage

The only required argument is `--data`. The script defaults to using `best.pt` from the project root as the base model and overwrites it with the fine-tuned result (after prompting for confirmation):

```bash
python finetune.py --data /path/to/my_dataset/data.yaml
```

### Full example with all options

```bash
python finetune.py \
  --data /path/to/my_dataset/data.yaml \
  --base-model best.pt \
  --output best.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

### CLI flag reference

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | *(required)* | Path to `data.yaml`. Must exist. |
| `--base-model` | `best.pt` (script dir) | Starting weights. Relative paths resolve from the script directory. |
| `--output` | `best.pt` (script dir) | Where to write the fine-tuned model. Defaults to overwriting `best.pt` in place. |
| `--epochs` | `50` | Number of training epochs. More epochs = longer training but potentially higher accuracy. |
| `--imgsz` | `640` | Input image size in pixels. Must match training resolution for best results. |
| `--batch` | `16` | Batch size. Reduce to `8` or `4` if you get CUDA out-of-memory errors. |
| `--device` | auto | Training device: `cpu`, `0` (first GPU), `cuda:0`, `mps` (Apple Silicon). |
| `--project` | `runs/train` | Parent directory where training run artifacts are saved. |
| `--name` | `finetune_run` | Subdirectory name under `--project` for this run's artifacts. |
| `--no-confirm` | off | Skip the overwrite confirmation prompt. Useful for automated pipelines. |

Omitting `--output` defaults to overwriting `best.pt`. The script will prompt:
```
This will overwrite the existing model at /path/to/best.pt. Continue? [y/N]:
```
Pass `--no-confirm` to skip this prompt.

---

## Step 4: Monitor Training

While training runs, ultralytics prints per-epoch metrics to the terminal:

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50     3.21G      1.742      2.145      1.231         48        640
  2/50     3.21G      1.631      1.983      1.189         52        640
...
```

- **box_loss** — bounding box regression loss (lower = better localization).
- **cls_loss** — classification loss (lower = fewer class errors).
- **mAP50 / mAP50-95** — mean average precision at IoU thresholds (higher = better detection quality). These are printed at validation intervals.

Raw training artifacts — including all epoch checkpoints, confusion matrices, and metric plots — are saved under:
```
runs/train/finetune_run/
```

The best checkpoint by validation mAP is saved automatically at:
```
runs/train/finetune_run/weights/best.pt
```

---

## Step 5: Replace the Model

The script automatically copies the best checkpoint to the `--output` path when training completes. You will see:
```
Fine-tuning complete. New model saved to: /path/to/best.pt
```

If you overwrote the live `best.pt` used by the web server, **restart the server** to load the new model:

```bash
uvicorn backend.app:app --reload
```

The YOLO model is loaded **lazily** on the first `/detect` call — it is not loaded at server startup. A reload is sufficient; no other configuration changes are needed.

---

## Step 6: Verify the New Model

Run a quick sanity check to confirm the new model has the correct number of classes:

```bash
python -c "from ultralytics import YOLO; m = YOLO('best.pt'); print(m.names)"
```

**Expected output:** a dict with exactly 11 entries, e.g.:
```python
{0: 'damage-front-windscreen', 1: 'damaged-door', 2: 'damaged-fender', ...}
```

If the count differs from 11, the model was trained with a different `nc` value in `data.yaml`. Retrain using the correct template from Step 2.

---

## Troubleshooting

**"CUDA out of memory" during training**
Reduce `--batch` to `8` or `4`. You can also reduce `--imgsz` to `320` at the cost of some accuracy.

**"No labels found" or training exits immediately with 0 samples**
Verify that your `labels/` directory mirrors your `images/` directory structure. Each image must have a corresponding `.txt` file with the same stem. Check that label files are not empty and contain valid normalized coordinates (all values between 0 and 1).

**Training completes but mAP is very low**
- Increase dataset size — aim for 50+ examples per class that you want to improve.
- Increase `--epochs` (try 100).
- Review a sample of label files to confirm bounding boxes are correct and class IDs are accurate.
- Ensure the train/val split is representative (similar camera angles and lighting conditions in both sets).

**"base model not found" error**
The script looks for `best.pt` in the project root (the directory containing `finetune.py`). If you moved the model, pass its location explicitly:
```bash
python finetune.py --data data.yaml --base-model /path/to/your/model.pt
```

**Script exits with "data.yaml not found"**
The `--data` path must point to the actual `data.yaml` file, not the dataset directory. Example:
```bash
# Correct:
python finetune.py --data /my_dataset/data.yaml

# Wrong:
python finetune.py --data /my_dataset
```

---

## Notes on Class Vocabulary

The 11 existing classes (`class_dict.txt`) must remain unchanged. The web app's `/detect` and `/ai-analysis` endpoints use these exact class names in YOLO inference and AI provider prompts. Changing the class list invalidates the existing `best.pt` and requires a complete retrain from scratch.

If you need to add new damage types (e.g., scratches, rust):
1. Update `class_dict.txt` with the new class names.
2. Update the AI prompt in `backend/app.py` to reference the new classes.
3. Collect and label a full training dataset for all classes (existing + new).
4. Retrain from a base YOLOv8 checkpoint (e.g., `yolov8n.pt`) rather than fine-tuning from the existing `best.pt`.
