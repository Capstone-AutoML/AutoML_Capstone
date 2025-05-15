import os
import json
import random
from pathlib import Path
from typing import List, Dict
import cv2

# --- Configuration ---
yolo_dir = Path("mock_io/data/prelabelled/yolo")
dino_dir = Path("mock_io/data/prelabelled/gdino")
labeled_dir = Path("mock_io/data/labeled")
mismatched_dir = Path("mock_io/data/mismatched/pending")
raw_image_dir = Path("mock_io/data/raw/images")
output_root = Path("mock_io/boxed_images")

# Ensure output root exists
output_root.mkdir(parents=True, exist_ok=True)

# --- Helper function: draw boxes on an image ---
def draw_boxes_on_image(image_path: Path, predictions: List[Dict], output_path: Path, label_source: str):
    image = cv2.imread(str(image_path))
    if image is None:
        return

    for pred in predictions:
        x1, y1, x2, y2 = map(int, pred["bbox"])
        class_name = pred["class"]
        confidence = pred.get("confidence", 0.0)
        label = f"{class_name} ({confidence:.2f})"

        color = (0, 255, 0) if label_source == "YOLO" else (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)

# --- Draw 10 random images from a folder ---
def sample_and_draw(json_dir: Path, output_subdir: str, label_source: str, sample_size: int = 10):
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

# --- Draw all images from a folder (for labeled) ---
def draw_all(json_dir: Path, output_subdir: str, label_source: str):
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

# --- Main execution ---
if __name__ == "__main__":
    print("Drawing bounding boxes...")

    sample_and_draw(yolo_dir, "yolo_outputs", "YOLO")
    sample_and_draw(dino_dir, "dino_outputs", "DINO")
    sample_and_draw(mismatched_dir, "mismatched_outputs", "YOLO")
    draw_all(labeled_dir, "labeled_outputs", "YOLO")

    print(f"Boxed images saved in: {output_root.resolve()}")
