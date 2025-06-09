# Quantization

This script performs model quantization to reduce model size and improve inference speed. It supports three quantization methods:

- **ONNX Dynamic Quantization**
- **FP16 Quantization**
- **IMX Post Training Quantization** (Linux only)

---

## Key Functions

### `quantize_model(model_path, quantize_config_path)`

Entry point for applying quantization. Based on the method specified in the configuration file, it routes the process to one of the supported techniques. It also prepares calibration data for IMX quantization if needed.

### `imx_quantization(model, output_path, quantize_yaml)`

Exports the model in IMX format using Ultralytics export function with `format="imx"`.

### `fp16_quantization(model, output_path)`

Converts model weights from FP32 to FP16 precision using PyTorch's built-in capabilities.

### `onnx_quantization(model, output_path, preprocessed_path)`

Converts YOLO model to ONNX format and applies dynamic quantization.

---

## Supported Methods

### 1. ONNX Quantization

```python
def onnx_quantization(model, output_path, preprocessed_path)
```

- **Use Case**: Cross-platform deployment with slightly reduced accuracy
- Converts YOLO model to ONNX format
- Preprocesses it using `onnxruntime` tools
- Applies dynamic quantization using `quantize_dynamic`

### 2. FP16 Quantization

```python
def fp16_quantization(model, output_path)
```

- **Use Case**: Fast conversion with minimal accuracy loss
- Converts model weights to FP16 precision
- Saves as a `.pt` PyTorch model

### 3. IMX Quantization (Linux only)

```python
def imx_quantization(model, output_path, quantize_yaml)
```

- **Use Case**: Sony IMX500 edge device deployment (e.g. Raspberry Pi AI Cameras)
- Requires Linux environment and Java installed (`sudo apt install openjdk-11-jdk`)
- Uses Ultralytics export function with `format="imx"`
- Requires a YAML config file with dataset paths and class information

---

## Configuration

The quantization process uses a configuration file (`src/quantize_config.json`) which must include the following:

- **`quantization_method`**: One of `"ONNX"`, `"FP16"`, or `"IMX"`
- **`output_dir`**: Path to save the quantized model
- **`labeled_json_path`**: Path to labeled data JSON files (required for IMX)
- **`labeled_images_path`**: Path to labeled images (required for IMX)
- **`quantization_data_path`**: Directory for calibration dataset (required for IMX)
- **`calibration_samples`**: Number of samples for IMX calibration (recommended: 300+)
- **`quantize_yaml_path`**: Path where YAML config will be created (required for IMX)

---

### Example Usage

```python
quantize_model(
    model_path="mock_io/model_registry/model/nano_trained_model.pt",
    quantize_config_path="src/quantize_config.json"
)
```

---

### Notes

- ONNX quantization is portable and works across platforms.
- FP16 is fast and requires minimal changes.
- IMX quantization is recommended for deployment on Linux edge devices and requires proper calibration data which is automatically prepared from `labeled_images_path`. IMX documentation recommends using 300+ images for optimal calibration.

---

### Errors to Watch For

- Missing `quantize_yaml_path`
- Running IMX quantization on a non-Linux system
- Invalid model path or corrupted weights

---

This module is a crucial step in optimizing YOLO models for real-time deployment, especially in resource-constrained environments.
