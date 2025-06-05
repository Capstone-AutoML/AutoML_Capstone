"""
Utility functions for the pipeline.
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd
from typing import Dict, Union, Optional
import json
import os
import torch
import shutil
import random
from PIL import Image


def detect_device() -> str:
    """
    Detect and return the best available device for model inference.
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict:
    """Load configuration parameters from a JSON config file.
    
    Parameters
    ----------
    config_path : Optional[Union[str, Path]], default=None
        Path to the configuration JSON file. If None, looks for 'config.json' in the current directory.
        
    Returns
    -------
    Dict
        Dictionary containing configuration parameters
        
    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist
    ValueError
        If distillation_image_prop is invalid (negative or > 1 when ratio)
    """
    if config_path is None:
        config_path = Path("config.json")
    else:
        config_path = Path(config_path)
        
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate distillation_image_prop if present
    if "distillation_image_prop" in config:
        prop = config["distillation_image_prop"]
        if isinstance(prop, (int, float)):
            if prop < 0:
                raise ValueError("distillation_image_prop cannot be negative")
            if 0 < prop < 1:  # Ratio
                if prop > 1:
                    raise ValueError("distillation_image_prop ratio cannot be greater than 1")
        else:
            raise ValueError("distillation_image_prop must be a number")
            
    return config


def draw_yolo_bboxes(img_path, label_path, label_map=None):
    """Draw YOLO-format bounding boxes on an image.

    Parameters
    ----------
    img_path : Path or str
        Path to the image file.
    label_path : Path or str
        Path to the corresponding YOLO label file.
    """
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    fig, ax = plt.subplots()
    ax.imshow(image)

    with open(label_path, "r") as f:
        for line in f:
            cls_id, x_center, y_center, width, height = map(float, line.strip().split())
            cls_id = int(cls_id)
            cls_name = ""
            if label_map is not None:
                cls_name = label_map[cls_id]
            x = (x_center - width / 2) * w
            y = (y_center - height / 2) * h
            box_w = width * w
            box_h = height * h

            rect = patches.Rectangle((x, y), box_w, box_h,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 5, f"{cls_id}: {cls_name}", color='red',
                    fontsize=10, backgroundcolor='white')

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def prepare_training_data(config: dict):
    """
    Splits dataset into train/val/test sets and saves the images and
    labels. Labels are converted from JSON to YOLO format. 
    True negative images with no labels are also added to the dataset.
    Args:
        config (dict): Dictionary containing required paths and optional split ratios.
                       Expected keys:
                       - "augmented_images_path"
                       - "augmented_labels_path"
                       - "true_negative_images_path"
                       - "training_output_path"
                       - "train_val_test_split" (list of 3 floats summing to 1.0)
    """
    # Define paths
    aug_images = Path(config["augmented_images_path"])
    aug_labels = Path(config["augmented_labels_path"])
    true_negatives = Path(config["true_negative_images_path"])
    out_dir = Path(config["training_output_path"])
    split_ratio = config.get("train_val_test_split", [0.7, 0.2, .10])  # default to train/val only

    # Validate split ratio
    assert abs(sum(split_ratio) - 1.0) < 1e-6, "Train/val/test split ratios must sum to 1.0"

    # Convert classes to int in a dictionary
    class_map = {
        "FireBSI": 0,
        "LightningBSI": 1,
        "PersonBSI": 2,
        "SmokeBSI": 3,
        "VehicleBSI": 4
    }

    image_label_pairs = []

    # Convert JSON labels to YOLO format
    for json_file in aug_labels.glob("*.json"):
        image_file = aug_images / (json_file.stem + ".jpg")
        if not image_file.exists():
            image_file = image_file.with_suffix(".png")
        if not image_file.exists():
            print(f"Image for {json_file.name} not found. Skipping.")
            continue

        with open(json_file) as f:
            data = json.load(f)

        yolo_lines = []
        im = Image.open(image_file)
        w, h = im.size

        for ann in data.get("predictions", []):
            cls = ann["class"]
            if cls not in class_map:
                print(f"Unknown class '{cls}' in {json_file.name}. Skipping annotation.")
                continue
            class_id = class_map[cls]

            # Convert [xmin, ymin, xmax, ymax] to YOLO format: https://medium.com/@telega.slawomir.ai/json-to-yolo-dataset-converter-9e9e643a31a7
            x_min, y_min, x_max, y_max = ann["bbox"]
            box_w = x_max - x_min
            box_h = y_max - y_min
            x_center = (x_min + box_w / 2) / w
            y_center = (y_min + box_h / 2) / h
            norm_w = box_w / w
            norm_h = box_h / h

            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")

        image_label_pairs.append((image_file, yolo_lines))

    # Add true negatives (images with no labels)
    for img_file in list(true_negatives.glob("*.jpg")) + list(true_negatives.glob("*.png")):
        image_label_pairs.append((img_file, []))

    # Shuffle and split data into train/val/test
    random.shuffle(image_label_pairs)
    n_total = len(image_label_pairs)
    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    train_pairs = image_label_pairs[:n_train]
    val_pairs = image_label_pairs[n_train:n_train + n_val]
    test_pairs = image_label_pairs[n_train + n_val:]

    # Save to YOLO-style structure
    for split_name, split_data in zip(["train", "val", "test"], [train_pairs, val_pairs, test_pairs]):
        for img_path, labels in split_data:
            dest_img = out_dir / "images" / split_name / img_path.name
            dest_lbl = out_dir / "labels" / split_name / (img_path.stem + ".txt")
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            dest_lbl.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(img_path, dest_img)
            with open(dest_lbl, "w") as f:
                f.write("\n".join(labels))

    print(f"[INFO] Training data prepared at '{out_dir}'")
    print(f"[INFO] Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")