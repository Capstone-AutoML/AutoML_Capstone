import os
import json
import random
from pathlib import Path
from typing import List, Dict
import cv2

# --- Configuration Directories ---

# Directories for input JSON annotations from different sources
# automl_workspace/data_pipeline/prelabeled
yolo_dir = Path("automl_workspace/data_pipeline/prelabeled/yolo")
dino_dir = Path("automl_workspace/data_pipeline/prelabeled/gdino")
labeled_dir = Path("automl_workspace/data_pipeline/labeled")
mismatched_dir = Path("automl_workspace/data_pipeline/label_sutdio/pending")

# Directory containing raw input images
raw_image_dir = Path("automl_workspace/data_pipeline/input")

# Root directory to store output images with bounding boxes
output_root = Path("automl_workspace/data_pipeline/boxed_images")

# Ensure output directory exists
output_root.mkdir(parents=True, exist_ok=True)

# --- Helper Function: Draw Bounding Boxes on Image ---

def draw_boxes_on_image(image_path: Path, predictions: List[Dict], output_path: Path, label_source: str):
    """
    Draws bounding boxes and class labels on an image and saves the result to the specified output path.

    Args:
        image_path (Path): Path to the raw image file (JPEG or PNG).
        predictions (List[Dict]): A list of dictionaries containing prediction details:
                                  each dictionary must have a 'bbox' (bounding box coordinates),
                                  a 'class' (label), and optionally a 'confidence' (float).
        output_path (Path): Path to save the output image with overlaid bounding boxes.
        label_source (str): Label source name (e.g., "YOLO" or "DINO"), used to determine box color.

    Returns:
        None
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred["bbox"])
        class_name = pred["class"]
        confidence = pred.get("confidence", 0.0)
        label = f"{class_name} ({confidence:.2f})"

        # Green for YOLO, red for other sources
        color = (0, 255, 0) if label_source == "YOLO" else (0, 0, 255)

        # Draw bounding box and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Ensure the output directory exists before saving
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

# --- Function: Draw Boxes on Sampled Images ---

def sample_and_draw(json_dir: Path, output_subdir: str, label_source: str, sample_size: int = 10):
    """
    Randomly samples a subset of JSON annotation files, overlays predicted bounding boxes on their corresponding images,
    and saves the results in the specified output subdirectory.

    Args:
        json_dir (Path): Directory containing JSON annotation files.
        output_subdir (str): Name of the subdirectory under the output root where results will be stored.
        label_source (str): Source of the labels, used for coloring (e.g., "YOLO", "DINO").
        sample_size (int): Number of JSON files to randomly sample. Default is 10.

    Returns:
        None
    """
    all_files = list(json_dir.glob("*.json"))
    sample_files = random.sample(all_files, min(len(all_files), sample_size))
    output_dir = output_root / output_subdir

    for json_file in sample_files:
        image_name = json_file.stem
        image_path = raw_image_dir / f"{image_name}.jpg"
        if not image_path.exists():
            image_path = raw_image_dir / f"{image_name}.png"
        if not image_path.exists():
            continue

        with open(json_file) as f:
            data = json.load(f)
            predictions = data.get("predictions", [])

        output_path = output_dir / f"{image_name}.jpg"
        draw_boxes_on_image(image_path, predictions, output_path, label_source)

# --- Function: Draw Boxes on All Images ---

def draw_all(json_dir: Path, output_subdir: str, label_source: str):
    """
    Processes all JSON annotation files in a given directory, draws predicted bounding boxes on corresponding images,
    and saves the results in the designated output subdirectory.

    Args:
        json_dir (Path): Directory containing JSON annotation files.
        output_subdir (str): Name of the subdirectory under the output root where results will be stored.
        label_source (str): Source of the labels, used for coloring (e.g., "YOLO", "DINO").

    Returns:
        None
    """
    all_files = list(json_dir.glob("*.json"))
    output_dir = output_root / output_subdir

    for json_file in all_files:
        image_name = json_file.stem
        image_path = raw_image_dir / f"{image_name}.jpg"
        if not image_path.exists():
            image_path = raw_image_dir / f"{image_name}.png"
        if not image_path.exists():
            continue

        with open(json_file) as f:
            data = json.load(f)
            predictions = data.get("predictions", [])

        output_path = output_dir / f"{image_name}.jpg"
        draw_boxes_on_image(image_path, predictions, output_path, label_source)

# --- Main Execution Block ---

if __name__ == "__main__":
    """
    Entry point of the script. Executes drawing functions to visualize predictions from different sources
    (YOLO, DINO, mismatched, and labeled) and saves annotated images to their respective directories.
    """
    print("Drawing bounding boxes...")

    sample_and_draw(yolo_dir, "yolo_outputs", "YOLO")
    sample_and_draw(dino_dir, "dino_outputs", "DINO")
    sample_and_draw(mismatched_dir, "mismatched_outputs", "YOLO")
    draw_all(labeled_dir, "labeled_outputs", "YOLO")

    print(f"Boxed images saved in: {output_root.resolve()}")
