# Matching Logic

This script compares YOLO and Grounding DINO predictions for the same image and flags mismatches for human review. It evaluates object matches based on class name and Intersection-over-Union (IoU) and applies configurable thresholds to determine the confidence of each detection.

## Overview

- **Input**:  
  - YOLO-generated JSON files  
  - DINO-generated JSON files

- **Output**:  
  - Matched files → saved to `labeled_dir`  
  - Mismatched files → saved to `pending_dir` with `label_status: 0`

- **Configurable Thresholds**:  
  - `iou_threshold`  
  - `low_conf_threshold`  
  - `mid_conf_threshold`  
  - `dino_false_negative_threshold`

---

## Functions

### `compute_iou(box1, box2)`
Computes the Intersection-over-Union between two bounding boxes.

- `box1`, `box2`: Lists of `[x1, y1, x2, y2]` format  
- **Returns**: `float` IoU score

---

### `normalize_class(c)`
Cleans up class names for matching purposes.

- Removes `"BSI"` suffixes and lowercases  
- **Returns**: normalized class name

---

### `match_predictions(yolo_preds, dino_preds, iou_thresh)`
Matches YOLO predictions to DINO predictions by class name and IoU.

- `yolo_preds`, `dino_preds`: Lists of prediction dictionaries  
- `iou_thresh`: Minimum IoU to consider a match  
- **Returns**: `List[bool]` indicating which YOLO predictions matched

---

### `match_and_filter(yolo_dir, dino_dir, labeled_dir, pending_dir, config)`
Main function to match predictions and split into labeled or pending sets.

- Loads predictions from YOLO and DINO
- Flags mismatches based on:
  - Low/medium YOLO confidence
  - High-confidence DINO detections missed by YOLO

#### Output Actions:
- Adds `label_status: 0` and `confidence_flag` to flagged predictions
- Saves:
  - Confident matches to `labeled_dir`
  - Mismatches to `pending_dir` for human review

#### Config Example:

```python
config = {
    "iou_threshold": 0.5,
    "low_conf_threshold": 0.3,
    "mid_conf_threshold": 0.6,
    "dino_false_negative_threshold": 0.5
}
```

#### Summary Output:
- Total successfully processed files  
- Skipped/unmatched files  
- Files that failed to process due to error

---

## Example Usage

```python
match_and_filter(
    yolo_dir=Path("data/predictions/yolo"),
    dino_dir=Path("data/predictions/dino"),
    labeled_dir=Path("data/labeled"),
    pending_dir=Path("data/mismatched/pending"),
    config=config
)
```
