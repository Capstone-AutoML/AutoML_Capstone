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
import yaml
import os
import torch
import numpy as np
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


def draw_yolo_bboxes(
    img: Union[str, Path, np.ndarray],
    label: Union[str, Path, np.ndarray],
    label_map: Optional[Dict[int, str]] = None
) -> None:
    """
    Draw YOLO-format bounding boxes on an image.
    
    Parameters
    ----------
    img : Union[str, Path, np.ndarray]
        Either a path to an image file or a numpy array of the image
    label : Union[str, Path, np.ndarray]
        Either a path to a YOLO label file or a 2D array of YOLO format boxes
        (each row: [class_id, x_center, y_center, width, height])
    label_map : Optional[Dict[int, str]], optional
        Dictionary mapping class IDs to class names, by default None
    """
    # Handle image input
    if isinstance(img, (str, Path)):
        image = cv2.imread(str(img))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = img.copy()
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
    
    h, w, _ = image.shape
    
    # Create figure
    fig, ax = plt.subplots()
    ax.imshow(image)
    
    # Handle label input
    if isinstance(label, (str, Path)):
        with open(label, "r") as f:
            boxes = [list(map(float, line.strip().split())) for line in f]
    else:
        boxes = label
    
    # Draw boxes
    for box in boxes:
        cls_id, x_center, y_center, width, height = box
        cls_id = int(cls_id)
        
        # Get class name if label_map is provided
        cls_name = label_map[cls_id] if label_map is not None else ""
        
        # Convert normalized coordinates to pixel coordinates
        x = (x_center - width / 2) * w
        y = (y_center - height / 2) * h
        box_w = width * w
        box_h = height * h
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x, y), box_w, box_h,
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        ax.text(
            x, y - 5,
            f"{cls_id}: {cls_name}",
            color='red',
            fontsize=10,
            backgroundcolor='white'
        )
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def create_data_yaml(output_dir: str, yaml_path: str = None):
    """
    Create a data.yaml file for training with absolute paths.

    Args:
        output_dir (str): Directory containing the training images and labels.
        yaml_path (str, optional):
            Output path for the YAML file.
            Defaults to "output_dir/data.yaml".
    """
    if yaml_path is None:
        yaml_path = os.path.join(output_dir, "data.yaml")

    # Resolve absolute paths
    train_path = os.path.abspath(os.path.join(output_dir, "images", "train"))
    val_path = os.path.abspath(os.path.join(output_dir, "images", "val"))
    test_path = os.path.abspath(os.path.join(output_dir, "images", "test"))

    # YAML content
    data_yaml = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
        "nc": 5,
        "names": ["FireBSI", "LightningBSI", "PersonBSI", "SmokeBSI", "VehicleBSI"]
    }

    # Write YAML file
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=None)

    print(f"[INFO] data.yaml saved to: {yaml_path}")


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

    # Create data.yaml file
    create_data_yaml(out_dir)


def prepare_quantization_data(config: dict, image_dir: Path, calibration_samples: int = 200):
    """
    Prepare a calibration dataset for IMX quantization by sampling from incoming data.

    Args:
        config (dict): Dictionary containing required paths.
                       Expected keys:
                       - "labeled_json_path" (original labeled data)
                       - "quantization_data_path" (where to save calibration data)
        image_dir (Path): Directory containing the labeled images to sample from.
        calibration_samples (int): Number of images to use for calibration (default: 200)
    """
    quant_dir = Path(config.get("quantization_data_path", "data/quantization"))

    # If calibration data already exists, use it
    if (quant_dir / "images" / "train").exists() and (quant_dir / "images" / "val").exists():
        print(f"[INFO] Using existing calibration data at '{quant_dir}'")
        print("[INFO] To recreate calibration data, delete the quantization directory")
        create_quantize_yaml(quant_dir)
        return

    print("[INFO] Creating new calibration dataset...")

    labeled_json_dir = Path(config.get("labeled_json_path", "mock_io/data/labeled"))

    if not labeled_json_dir.exists():
        raise FileNotFoundError(f"Labeled JSON directory not found at {labeled_json_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found at {image_dir}")

    # Convert classes to int in a dictionary (same as training)
    class_map = {
        "FireBSI": 0,
        "LightningBSI": 1,
        "PersonBSI": 2,
        "SmokeBSI": 3,
        "VehicleBSI": 4
    }

    all_image_label_pairs = []

    # Process JSON labels to YOLO format using the provided image_dir
    json_files = list(labeled_json_dir.glob("*.json"))
    image_files = [f for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    image_lookup = {f.stem.lower(): f for f in image_files}

    # Match JSON files to images
    matched_pairs = [
        (json_file, image_lookup[json_file.stem.lower()])
        for json_file in json_files
        if json_file.stem.lower() in image_lookup
    ]

    print(f"[INFO] Matched {len(matched_pairs)} JSON-image pairs")

    for json_file, image_file in matched_pairs:
        with open(json_file) as f:
            data = json.load(f)

        yolo_lines = []
        im = Image.open(image_file)
        w, h = im.size

        # Only process images that have predictions
        predictions = data.get("predictions", [])
        if not predictions:
            continue

        for ann in predictions:
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

        if yolo_lines:
            all_image_label_pairs.append((image_file, yolo_lines))

    print(f"[INFO] Found {len(all_image_label_pairs)} images with valid annotations for calibration")

    # Stratified sampling to ensure class representation
    class_images = {i: [] for i in range(5)}
    for img_path, labels in all_image_label_pairs:
        classes_in_image = set()
        for label in labels:
            class_id = int(label.split()[0])
            classes_in_image.add(class_id)
        for class_id in classes_in_image:
            class_images[class_id].append((img_path, labels))

    for class_id in range(5):
        print(f"[INFO] Class {class_id}: {len(class_images[class_id])} images available")

    # Sample from each class proportionally
    selected_pairs = []
    samples_per_class = calibration_samples // 5

    for class_id in range(5):
        class_samples = class_images[class_id]
        if class_samples:
            n_samples = min(samples_per_class, len(class_samples))
            if n_samples > 0:
                sampled = random.sample(class_samples, n_samples)
                selected_pairs.extend(sampled)
                print(f"[INFO] Sampled {len(sampled)} images for class {class_id}")
            else:
                print(f"[WARNING] No samples available for class {class_id}")

    # Remove duplicates
    seen_images = set()
    unique_pairs = []
    for img_path, labels in selected_pairs:
        img_name = img_path.name
        if img_name not in seen_images:
            seen_images.add(img_name)
            unique_pairs.append((img_path, labels))

    selected_pairs = unique_pairs[:calibration_samples]
    random.shuffle(selected_pairs)

    print(f"[INFO] Selected {len(selected_pairs)} unique images for IMX calibration")

    # Create IMX quantization dataset structure
    calib_train_ratio = 0.8
    n_train = int(len(selected_pairs) * calib_train_ratio)

    train_pairs = selected_pairs[:n_train]
    val_pairs = selected_pairs[n_train:]

    # Save calibration dataset
    for split_name, split_data in zip(["train", "val"], [train_pairs, val_pairs]):
        for img_path, labels in split_data:
            dest_img = quant_dir / "images" / split_name / img_path.name
            dest_lbl = quant_dir / "labels" / split_name / (img_path.stem + ".txt")
            dest_img.parent.mkdir(parents=True, exist_ok=True)
            dest_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, dest_img)
            with open(dest_lbl, "w") as f:
                f.write("\n".join(labels))

    print(f"[INFO] IMX calibration data prepared at '{quant_dir}'")
    print(f"[INFO] Calibration Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # Create quantize.yaml file for IMX
    create_quantize_yaml(quant_dir)


def create_distill_yaml(output_dir: str, yaml_path: str = None):
    """
    Create a distillation_data.yaml file for distillation with absolute paths.

    Args:
        output_dir (str): Directory containing the distillation images and labels.
        yaml_path (str, optional): Output path for the YAML file.
                                  Defaults to "mock_io/data/distillation/distillation_data.yaml".
    """
    if yaml_path is None:
        yaml_path = "mock_io/data/distillation/distillation_data.yaml"

    # Resolve absolute paths
    train_path = os.path.abspath(os.path.join(output_dir, "train"))
    val_path = os.path.abspath(os.path.join(output_dir, "valid"))

    # YAML content
    distill_yaml = {
        "train": train_path,
        "val": val_path,
        "nc": 5,
        "names": ["FireBSI", "LightningBSI", "PersonBSI", "SmokeBSI", "VehicleBSI"]
    }

    # Write YAML file
    with open(yaml_path, "w") as f:
        yaml.dump(distill_yaml, f, sort_keys=False, default_flow_style=None)

    print(f"[INFO] distillation_data.yaml saved to: {yaml_path}")


def create_quantize_yaml(output_dir: str, yaml_path: str = None):
    """
    Create a quantize.yaml file for IMX quantization with absolute paths.

    Args:
        output_dir (str): Directory containing the calibration images and labels.
        yaml_path (str, optional): Output path for the YAML file.
                                  Defaults to "src/quantize.yaml".
    """
    if yaml_path is None:
        yaml_path = "src/quantize.yaml"

    # Resolve absolute paths (no test path for quantization)
    train_path = os.path.abspath(os.path.join(output_dir, "images", "train"))
    val_path = os.path.abspath(os.path.join(output_dir, "images", "val"))

    # YAML content
    quantize_yaml = {
        "train": train_path,
        "val": val_path,
        "nc": 5,
        "names": ["FireBSI", "LightningBSI", "PersonBSI", "SmokeBSI", "VehicleBSI"]
    }

    # Write YAML file
    with open(yaml_path, "w") as f:
        yaml.dump(quantize_yaml, f, sort_keys=False, default_flow_style=None)

    print(f"[INFO] quantize.yaml saved to: {yaml_path}")
