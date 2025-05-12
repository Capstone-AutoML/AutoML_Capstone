"""
Script for automated pre-labelling using SAM with YOLO box prompts.
Each detected object (from YOLO) is passed as a bounding box to SAM,
which generates one mask per object and saves them as individual PNGs.
"""

from pathlib import Path
from typing import Dict, List
import sys
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import SAM

# Add parent directory to path to import utils (if needed)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files from a directory.
    """
    return [f for f in directory.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

def _load_model(model_path: Path) -> SAM:
    """
    Load the SAM model from the given path.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"SAM model not found at {model_path}")
    return SAM(str(model_path))

def _save_metadata(metadata: List[Dict], output_path: Path) -> None:
    """
    Save mask metadata as JSON.
    """
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

def generate_segmentation(
    raw_dir: Path,
    yolo_json_dir: Path,
    mask_output_dir: Path,
    metadata_output_dir: Path,
    model_path: Path,
    config: Dict,
    verbose: bool = False
) -> None:
    """
    Run SAM prelabelling using YOLO box prompts.
    Saves one binary mask image per detected object,
    and JSON metadata per image.
    """
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_output_dir.mkdir(parents=True, exist_ok=True)

    sam_model = _load_model(model_path)
    print(f"Loaded SAM model from {model_path}")

    low_thresh = config.get("low_conf_threshold", 0.1)
    image_files = _get_image_files(raw_dir)
    print(f"Found {len(image_files)} images to process")

    successful, failed = 0, 0

    for image_path in tqdm(image_files, desc="Processing SAM masks"):
        image_name = image_path.stem
        json_path = yolo_json_dir / f"{image_name}.json"

        if not json_path.exists():
            failed += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            failed += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(json_path) as f:
            predictions = json.load(f).get("predictions", [])

        metadata = []
        counter = 0

        for pred in predictions:
            conf = pred["confidence"]
            if conf < low_thresh:
                continue

            label = pred["class"]
            box = pred["bbox"]

            try:
                results = sam_model(image_rgb, bboxes=[box])
                masks = results[0].masks

                if masks is None or masks.data is None:
                    continue

                for mask_tensor in masks.data:
                    mask = mask_tensor.cpu().numpy()
                    if mask.ndim != 2 or mask.sum() == 0:
                        continue

                    mask_uint8 = (mask * 255).astype(np.uint8)
                    mask_name = f"{image_name}_mask_{counter}.png"
                    mask_path = mask_output_dir / mask_name

                    if not cv2.imwrite(str(mask_path), mask_uint8):
                        print(f"Failed to save mask: {mask_path}")
                        continue

                    metadata.append({
                        "image": image_name,
                        "bbox": box,
                        "class": label,
                        "confidence": conf,
                        "mask_path": str(mask_path.relative_to(mask_output_dir.parent)),
                        "masked": True
                    })
                    counter += 1

            except Exception as e:
                print(f"Error processing {image_name} box {box}: {e}")

        if metadata:
            out_path = metadata_output_dir / f"{image_name}.json"
            _save_metadata(metadata, out_path)
            successful += 1
        else:
            failed += 1

    print(f"\nSAM Prelabelling Summary:")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
