# Training

This page outlines the unit test coverage for the `train.py` module, which handles loading model training configurations and selecting the most recent trained model based on filename timestamps.

---

## Coverage Overview

The tests focus on the following functions:

- `load_train_config`: Loads and validates a training configuration file.
- `find_latest_model`: Identifies the latest YOLO model file based on naming pattern or falls back to a default model.

---

## Function Tests

### `load_train_config`

- `test_load_valid_config`:  
  Ensures a valid training config JSON file loads correctly and includes all expected keys and values.

- `test_missing_file_raises_error`:  
  Confirms that a missing config file raises a `FileNotFoundError`.

- `test_missing_required_keys`:  
  Validates that missing required config fields trigger an `AssertionError`.

---

### `find_latest_model`

- `test_returns_latest_model`:  
  Identifies the most recent YOLO model using timestamped filenames matching the pattern `*_updated_yolo.pt`.

- `test_returns_fallback_if_no_model`:  
  Returns the fallback path if no matching model is found in the directory.

- `test_model_pattern_only_matches_updated`:  
  Verifies that only files with `_updated_yolo.pt` are considered valid candidates for latest model selection.

---

## Summary

This page outlines a focused and reliable test suite that ensures:

- Training configuration files are present, valid, and complete
- Only appropriately named model files are selected for retraining or fine-tuning
- Robust fallback behavior in the absence of model artifacts

These tests help safeguard the training pipeline from misconfiguration or missing artifacts, ensuring repeatable and predictable training runs.
