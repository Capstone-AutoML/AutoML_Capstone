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
    