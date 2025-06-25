# Quantization

This page outlines the unit test coverage for the `quantization.py` module, which implements three quantization strategies (IMX, FP16, and ONNX) for YOLO models, including model preparation, export, and configuration management.

---

## Coverage Overview

The tests cover:

- IMX quantization logic (platform-dependent)
- FP16 quantization success and failure paths
- ONNX export and quantization with subprocess
- Fallback behavior for unsupported or misconfigured cases
- The complete `quantize_model()` dispatcher logic

---

## Fixtures

### `mock_model`  
Returns a mock YOLO model with `.export()`, `.save()`, and `.half()` methods stubbed for isolated testing.

### `sample_quantize_config`  
Creates a dummy config dictionary with values for output paths, methods, and calibration settings used in dispatcher tests.

---

## Function Tests

### `imx_quantization()`

- `test_imx_quantization_non_linux`:  
  Ensures IMX quantization exits cleanly on unsupported platforms like Windows.

- `test_imx_quantization_success`:  
  Simulates successful export and model file movement on a Linux environment using mocks and patches.

---

### `fp16_quantization()`

- `test_fp16_quantization_success`:  
  Confirms the model is saved as expected when no errors occur.

- `test_fp16_quantization_failure`:  
  Handles exceptions during saving and returns `None` when failure occurs.

---

### `onnx_quantization()`

- `test_onnx_quantization_success`:  
  Validates ONNX export and quantization pipeline, including subprocess calls and dynamic quantization.

---

### `quantize_model()`

- `test_quantize_model_imx`:  
  Simulates an end-to-end quantization call using the IMX method with config and YAML preparation patched.

- `test_quantize_model_unsupported_method`:  
  Verifies that unsupported quantization types raise a `ValueError`.

- `test_quantize_model_missing_yaml`:  
  Ensures an error is raised when the quantization YAML file is missing from the config.

---

## Summary

This test suite confirms robust handling of edge cases and method-specific branches in model quantization. It validates export success paths, platform constraints, and configuration-based logic for:

- IMX
- FP16
- ONNX  
