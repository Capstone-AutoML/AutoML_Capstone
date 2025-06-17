# Grounding DINO Prelabeling

This script performs object detection using a Grounding DINO model guided by text prompts. It processes a folder of raw images and outputs structured JSON files for each image containing detected object metadata.

## Overview

- **Input**: Directory of raw images
- **Output**: JSON files with bounding boxes, confidence scores, and class labels
- **Model**: Grounding DINO
- **Prompts**: `"fire"`, `"smoke"`, `"person"`, `"vehicle"`, `"lightning"`
- **Thresholds**: Box confidence and text match thresholds
- **Execution**: Runs sequentially on a specified device

## Constants

- `TEXT_PROMPTS`: Default object classes to detect
- `BOX_THRESHOLD`: Minimum bounding box confidence (default: `0.3`)
- `TEXT_THRESHOLD`: Minimum text-prompt alignment confidence (default: `0.25`)

---

## Functions

### `_get_image_files(directory)`
Scans a directory for `.jpg`, `.jpeg`, and `.png` files.

- `directory (Path)`: Path to image directory
- **Returns**: `List[Path]` of image files

---

### `generate_gd_prelabelling(raw_dir, output_dir, config, model_weights, config_path, text_prompts=TEXT_PROMPTS, box_threshold=None, text_threshold=None)`
Main function that runs Grounding DINO to detect prompted classes in images.

- `raw_dir (Path)`: Directory containing input images  
- `output_dir (Path)`: Output folder for predictions  
- `config (Dict)`: Dictionary with device and threshold options  
- `model_weights (Path)`: Path to DINO model checkpoint  
- `config_path (Path)`: Path to DINO config file  
- `text_prompts (List[str])`: List of classes to detect (default: predefined)  
- `box_threshold (float)`: Detection threshold (can be overridden via config)  
- `text_threshold (float)`: Text alignment threshold (can be overridden via config)

#### Output per image
- Image name
- Class label
- Bounding box: `[x1, y1, x2, y2]`
- Confidence score
- Source tag: `"grounding_dino"`

#### Summary Output
- Number of successful detections
- Number of skipped files (unreadable)
- Number of failed detections

---

## Configuration Parameters (from `pipeline_config.json`)

The following fields from the configuration file directly control **Grounding DINO’s** behavior:

 | **Key**                         | **Description**                                                                 |
 |--------------------------------|---------------------------------------------------------------------------------|
 | `torch_device`                 | Device to run the model on (`"cpu"` or `"cuda"`).                               |
 | `dino_box_threshold`           | Minimum confidence required for bounding boxes to be retained (default: `0.3`). |
 | `dino_text_threshold`          | Minimum alignment confidence between text prompt and region (default: `0.25`).  |
 | `dino_false_negative_threshold`| Confidence threshold to flag potential false negatives for review (default: `0.5`). |
These values can be overridden or adjusted in the configuration dictionary passed to the function.

---

## Example Usage

```python
generate_gd_prelabelling(
    raw_dir=Path("data/raw/images"),
    output_dir=Path("data/predictions/dino"),
    config={
        "torch_device": "cuda",
        "dino_box_threshold": 0.3,
        "dino_text_threshold": 0.25
    },
    model_weights=Path("weights/groundingdino.pth"),
    config_path=Path("configs/GroundingDINO_SwinT_config.py")
)
