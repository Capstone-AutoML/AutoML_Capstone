# Matching Logic

This page outlines the unit test coverage for the `matching.py` script, which compares YOLO and Grounding DINO predictions and routes files into `labeled` or `pending` folders based on IoU, confidence, and class agreement.

---

## Coverage Overview

This test suite validates the following core functions:

- `compute_iou`: Intersection-over-Union between two bounding boxes
- `normalize_class`: Standardizes class names (e.g., removes suffixes)
- `match_predictions`: Compares YOLO and DINO predictions using class name and IoU
- `match_and_filter`: Reads YOLO/DINO JSON predictions and splits them into labeled or pending output folders using thresholds

---

## Unit Test Descriptions

### `compute_iou`

- **`test_compute_iou_overlap`**  
  Validates IoU is correctly calculated for overlapping boxes.

- **`test_compute_iou_no_overlap`**  
  Ensures IoU returns 0 for non-overlapping boxes.

---

### `normalize_class`

- **`test_normalize_class_strips_suffix`**  
  Checks normalization of class strings (e.g., lowercasing, suffix removal).

---

### `match_predictions`

- **`test_match_predictions_positive`**  
  Confirms that two bounding boxes with high IoU and same class are matched.

- **`test_match_predictions_negative_iou`**  
  Confirms that boxes with low IoU do not match even if class names are the same.

---

### `match_and_filter`

All tests below use the `match_dirs` fixture, which sets up temporary `yolo`, `dino`, `labeled`, and `pending` directories for testing.

- **`test_match_and_filter_matched_goes_labeled`**  
  Valid match (class + IoU + confidence) goes to `labeled`.

- **`test_match_and_filter_low_conf_goes_pending`**  
  Low confidence YOLO prediction results in pending output.

- **`test_dino_false_negative_goes_pending`**  
  High-confidence DINO detection missed by YOLO triggers pending file.

- **`test_match_and_filter_invalid_yolo_json_skipped`**  
  Handles bad JSON gracefully — file is skipped, not labeled or pending.

---

## Configuration Parameters Used

These thresholds are passed as a dictionary to `match_and_filter` and influence labeling decisions:

| Key                         | Description |
|----------------------------|-------------|
| `iou_threshold`            | Minimum IoU for a match (e.g., `0.5`) |
| `low_conf_threshold`       | YOLO predictions below this are flagged as false positives |
| `mid_conf_threshold`       | Intermediate YOLO confidence triggers review |
| `dino_false_negative_threshold` | DINO detections above this are considered missed by YOLO |

---

## Summary

This page outlines robust coverage of the `matching.py` logic, ensuring:

- Reliable IoU calculation  
- Consistent class name normalization  
- Correct behavior across all matching outcomes  
- Fault tolerance for malformed files  
- Accurate routing of predictions based on configurable thresholds

Together, these tests ensure that only confident, well-aligned predictions are passed automatically to the labeled dataset, while edge cases and uncertainties are flagged for human review.
