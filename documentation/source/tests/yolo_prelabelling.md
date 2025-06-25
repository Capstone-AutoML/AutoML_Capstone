# YOLO Prelabeling

This page outlines the test coverage for the `yolo_prelabelling.py` script, which automates object detection using a YOLO model. The tests validate individual components as well as end-to-end behavior under a variety of conditions, including edge cases and failure scenarios.

---

## Coverage Overview

This test suite includes checks for:

- Model loading and device handling
- Image discovery and validation
- Inference result parsing
- JSON output generation
- Pipeline behavior with:
  - Empty inputs
  - Corrupted images
  - Missing models
  - Invalid paths
  - Verbose logging

---

## Fixtures

### `patch_yolo_model`
Mocks the `YOLO` constructor and inference behavior with a dummy model returning fixed predictions.

### `tmp_dirs_with_images`
Creates a temporary raw folder with:
- Two valid image files
- One corrupted image file
- One non-image file  
Also creates an empty output directory.

### `mock_model_path`
Creates a temporary dummy model file (`model.pt`) to simulate a valid model checkpoint.

---

## Unit Test Descriptions

### `_get_image_files`

- **`test_get_image_files_valid`**: Returns only image files, including corrupted ones (based on extension).
- **`test_get_image_files_empty_directory`**: Returns an empty list when no files exist.

---

### `_load_model`

- **`test_load_model_success`**: Verifies successful model loading and `.to(device)` usage.
- **`test_load_model_file_not_found`**: Raises `FileNotFoundError` when the model file is missing.

---

### `_process_prediction`

- **`test_process_prediction`**: Parses a mock YOLO result into properly structured predictions.
- **`test_process_prediction_empty`**: Returns an empty list when no boxes are present.

---

### `_save_predictions`

- **`test_save_predictions`**: Writes predictions to a JSON file and verifies file structure.

---

### `generate_yolo_prelabelling`

- **`test_generate_yolo_prelabelling_success`**: Processes valid and corrupted images using the mock model and checks for correct output.
- **`test_generate_yolo_prelabelling_device_auto`**: Ensures fallback device detection works when `torch_device = "auto"`.
- **`test_generate_yolo_prelabelling_handles_corrupted_images`**: Ensures pipeline robustness against unreadable images.
- **`test_generate_yolo_prelabelling_model_error`**: Raises `FileNotFoundError` when model path is invalid.
- **`test_generate_yolo_prelabelling_empty_directory`**: Gracefully handles empty input folders.
- **`test_generate_yolo_prelabelling_verbose_mode`**: Captures and checks verbose log output.
- **`test_generate_yolo_prelabelling_creates_output_directory`**: Automatically creates output folder if it does not exist.
- **`test_generate_yolo_prelabelling_pytorch_mps_fallback`**: Verifies MPS fallback is enabled via the environment variable on Apple Silicon.

---

## Summary

This test suite ensures that `yolo_prelabelling.py` behaves correctly in:

- Normal workflows (valid images + model)
- OS-level and file-level failures
- Edge scenarios like:
  - Empty or missing folders
  - Corrupted or unsupported files
  - Unavailable device or model resources

The use of mocks ensures isolation from external dependencies while maintaining high test coverage and robustness.