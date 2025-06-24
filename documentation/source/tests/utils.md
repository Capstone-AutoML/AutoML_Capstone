# Utility Functions

This page outlines the unit test coverage for the `utils.py` module, which provides utility functions for configuration management, device detection, and YAML file generation.

---

## Coverage Overview

The following utilities are tested:

- `detect_device()`: Determines available hardware (CUDA, MPS, or CPU)
- `load_config()`: Loads and validates configuration from a JSON file
- `create_data_yaml()`: Generates a standard `data.yaml` for YOLO training
- `create_quantize_yaml()`: Creates a `quantize.yaml` file with placeholder values

---

## Function Tests

### Device Detection

- `test_detect_device_cuda`: Asserts CUDA is returned when available.
- `test_detect_device_mps`: Asserts MPS is returned if CUDA is unavailable.
- `test_detect_device_cpu`: Falls back to CPU when neither CUDA nor MPS is available.

### Configuration Loading

- `test_load_config_valid_file`: Loads a valid JSON config file and checks its content.
- `test_load_config_file_not_found`: Raises `FileNotFoundError` for a missing file.
- `test_load_config_invalid_json`: Raises `JSONDecodeError` for malformed JSON.
- `test_load_config_invalid_distillation_prop`: Raises `ValueError` if `distillation_image_prop` is negative.
- `test_load_config_default_path`: Validates fallback behavior using a mocked default path.

### YAML File Creation

- `test_create_data_yaml`: Asserts creation and content of a valid `data.yaml` file.
- `test_create_quantize_yaml`: Confirms `quantize.yaml` file is created in the expected location with placeholder structure.

---

## Summary

These unit tests ensure that utility functions used across the AutoML pipeline behave predictably and handle edge cases gracefully. They support robustness in configuration, hardware compatibility, and file generation workflows.
