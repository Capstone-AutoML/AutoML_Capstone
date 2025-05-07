"""
Utility functions for the pipeline.
"""

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd
from typing import Dict

def load_config() -> Dict:
    """
    TODO: Load configuration parameters from config file.
    """
    return {}


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
    