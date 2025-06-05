# YOLO Prelabeling

This script automates the generation of object detection annotations using a YOLO model. It loads a trained YOLO model, runs inference on all images in a given directory, and saves predictions in standardized JSON format.

## Overview

- **Input**: Directory of raw images
- **Output**: JSON files with bounding boxes, confidence scores, and class labels
- **Model**: YOLO (from `ultralytics`)
- **Device Handling**: Automatically selects CPU/GPU/MPS
- **Logging**: Summary of processed/failed images

## Functions

### `_load_model(model_path, device)`
Loads the YOLO model and moves it to the specified device.

- `model_path (Path)`: Path to the YOLO weights
- `device (str)`: Device name (`cpu`, `cuda`, or `mps`)
- **Returns**: `YOLO` model

---

### `_get_image_files(directory)`
Scans a directory for `.jpg`, `.jpeg`, and `.png` files.

- `directory (Path)`: Path to image directory
- **Returns**: `List[Path]` of image files

---

### `_process_prediction(result)`
Processes a YOLO result object into a structured list of prediction dictionaries.

- `result`: YOLO model inference result
- **Returns**: `List[Dict]` with keys: `bbox`, `confidence`, `class`

---

### `_save_predictions(predictions, output_path)`
Saves predictions to a JSON file.

- `predictions (List[Dict])`: Inference output
- `output_path (Path)`: Destination `.json` file

---

### `generate_yolo_prelabelling(raw_dir, output_dir, model_path, config, verbose=False)`
Main function to process a folder of raw images using the YOLO model.

- `raw_dir (Path)`: Directory containing input images
- `output_dir (Path)`: Output folder for JSON files
- `model_path (Path)`: Path to YOLO weights
- `config (Dict)`: Configuration including device
- `verbose (bool)`: Whether to log each processed file

#### Summary Output
- Number of images processed
- Number of failed images (with error messages)

## Example Usage

```python
generate_yolo_prelabelling(
    raw_dir=Path("data/raw/images"),
    output_dir=Path("data/predictions/yolo"),
    model_path=Path("models/yolo.pt"),
    config={"torch_device": "auto"},
    verbose=True
)
