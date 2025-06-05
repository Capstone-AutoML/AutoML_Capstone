import os
import json
from pathlib import Path
from typing import List, Dict
import shutil
from shapely.geometry import box as shapely_box


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two bounding boxes in [x1, y1, x2, y2] format.
    """
    b1 = shapely_box(*box1)
    b2 = shapely_box(*box2)
    return b1.intersection(b2).area / b1.union(b2).area


def normalize_class(c: str) -> str:
    """
    Normalize class names by removing BSI suffixes and lowercasing.
    Used for matching between YOLO and DINO predictions.
    """
    return c.lower().replace("bsi", "").strip()


def match_predictions(yolo_preds: List[Dict], dino_preds: List[Dict], iou_thresh: float) -> List[bool]:
    """
    Match YOLO boxes with DINO boxes using both class name and IOU.
    Returns a list of booleans indicating whether each YOLO box was matched.
    """
    matched = [False] * len(yolo_preds)
    used_dino = set()

    for i, yolo_obj in enumerate(yolo_preds):
        box_y = yolo_obj["bbox"]
        class_y = normalize_class(yolo_obj["class"])

        for j, dino_obj in enumerate(dino_preds):
            if j in used_dino:
                continue

            box_d = dino_obj["bbox"]
            class_d = normalize_class(dino_obj["class"])

            if class_y == class_d and compute_iou(box_y, box_d) >= iou_thresh:
                matched[i] = True
                used_dino.add(j)
                break

    return matched


def match_and_filter(
    yolo_dir: Path,
    dino_dir: Path,
    labeled_dir: Path,
    pending_dir: Path,
    config: Dict
) -> None:
    """
    Compare YOLO and DINO predictions for each image.
    Save YOLO prediction files to `labeled_dir` if they match DINO confidently.
    Save unmatched/mismatched files to `pending_dir` for review.

    Thresholds and behavior are controlled via config dictionary.
    """
    # Load threshold values from config
    iou_thresh = config.get("iou_threshold", 0.5)
    low_conf = config.get("low_conf_threshold", 0.3)
    mid_conf = config.get("mid_conf_threshold", 0.6)
    dino_fn_conf_thresh = config.get("dino_false_negative_threshold", 0.5)

    # Ensure output directories exist
    labeled_dir.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)

    # Track summary stats
    success_count = 0
    skipped_count = 0
    error_count = 0

    # Loop over YOLO JSON files
    for yolo_file in yolo_dir.glob("*.json"):
        filename = yolo_file.name
        dino_file = dino_dir / filename

        if not dino_file.exists():
            skipped_count += 1
            continue

        try:
            # Load YOLO and DINO predictions
            with open(yolo_file) as f:
                yolo_data = json.load(f)

            with open(dino_file) as f:
                dino_data = json.load(f)

            yolo_preds = yolo_data.get("predictions", [])
            dino_preds = dino_data.get("predictions", [])

            # Step 1: Match each YOLO box to DINO box using class + IOU
            matched_flags = match_predictions(yolo_preds, dino_preds, iou_thresh)

            # Step 2: Collect unmatched YOLO predictions
            unmatched = [
                pred for pred, matched in zip(yolo_preds, matched_flags) if not matched
            ]
            is_mismatch = False

            # Step 3: Handle unmatched YOLO detections using confidence thresholds
            for obj in unmatched:
                conf = obj["confidence"]
                cls = normalize_class(obj["class"])

                if conf < low_conf:
                    # Likely false positive
                    is_mismatch = True
                elif conf <= mid_conf:
                    # Uncertain, needs human review
                    is_mismatch = True
                # If conf > mid_conf → trusted YOLO prediction, do not flag

            # Step 4: Check for unmatched high-confidence DINO detections (false negatives)
            for dino_obj in dino_preds:
                cls_d = normalize_class(dino_obj["class"])
                is_matched = any(
                    normalize_class(y["class"]) == cls_d and compute_iou(y["bbox"], dino_obj["bbox"]) >= iou_thresh
                    for y in yolo_preds
                )
                if not is_matched and dino_obj["confidence"] > dino_fn_conf_thresh:
                    is_mismatch = True

            # Step 5: Save result
            if is_mismatch:
                # Add label_status = 0 to each YOLO object for review
                for obj in yolo_preds:

                    conf = obj["confidence"]
                    if conf < low_conf:
                        obj["confidence_flag"] = "low"
                    elif conf <= mid_conf:
                        obj["confidence_flag"] = "mid"
                    else:
                        obj["confidence_flag"] = "high"

                with open(pending_dir / filename, "w") as f:
                    json.dump({"predictions": yolo_preds}, f, indent=2)
            else:
                # If all objects match confidently → keep as labeled
                shutil.copyfile(yolo_file, labeled_dir / filename)

            success_count += 1

        except Exception:
            error_count += 1
            continue

    # Print summary
    print("\nPrediction Summary:")
    print(f"Successfully processed: {success_count} images")
    print(f"Skipped (unreadable): {skipped_count} images")
    print(f"Failed to process: {error_count} images")
