# Quantization

This script performs model quantization to reduce model size and improve inference speed. It supports three quantization methods:

- **ONNX Dynamic Quantization**
- **FP16 Quantization**
- **IMX Post Training Quantization** (Linux only)

---

## IMX Quantization Issues

Due to ongoing compatibility issues between required packages (such as `imx500-converter`, `uni-pytorch`, and `model-compression-toolkit`), we are currently unable to support IMX quantization in this pipeline. The dependency conflicts make it impossible to install all necessary packages together in a single environment, and attempts to resolve this automatically have not been successful.

We are temporarily relying on the ONNX Dynamic Quantization and FP16 Quantization options for model export and deployment. If you require IMX quantization, you may need to experiment with manual package pinning or use a separate, isolated environment.

- See: [Sony IMX500 Export for Ultralytics YOLO](https://docs.ultralytics.com/integrations/sony-imx500/)

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

Configure quantization in `automl_workspace/config/pipeline_config.json`:

```json
"process_options": {
  "skip_quantization": false  // Set to true to skip quantization
}
```

The quantization process uses a configuration file (`automl_workspace/config/quantize_config.json`) which must include the following:

- **`quantization_method`**: One of `"ONNX"`, `"FP16"`, or `"IMX"`
- **`output_dir`**: Path to save the quantized model
- **`labeled_json_path`**: Path to labeled data JSON files (required for IMX)
- **`labeled_images_path`**: Path to labeled images (required for IMX)
- **`quantization_data_path`**: Directory for calibration dataset (required for IMX)
- **`calibration_samples`**: Number of samples for IMX calibration (recommended: 300+)
- **`quantize_yaml_path`**: Path where YAML config will be created (required for IMX)

---

## Example Usage

**Pipeline Integration (Recommended):**

```json
// Configure in pipeline_config.json
"process_options": {
  "skip_quantization": false
}
```

```bash
# Run main pipeline
python src/main.py
```

**Standalone Usage:**

```python
from pipeline.quantization import quantize_model

quantized_path = quantize_model(
    model_path="automl_workspace/model_registry/model/nano_trained_model.pt",
    quantize_config_path="automl_workspace/config/quantize_config.json"
)
```

---

## Notes

- ONNX quantization is portable and works across platforms.
- FP16 is fast and requires minimal changes.
- IMX quantization is recommended for deployment on Linux edge devices and requires proper calibration data which is automatically prepared from `labeled_images_path`. IMX documentation recommends using 300+ images for optimal calibration.

---

## Errors to Watch For

- Missing `quantize_yaml_path`
- Running IMX quantization on a non-Linux system
- Invalid model path or corrupted weights

---

This module is a crucial step in optimizing YOLO models for real-time deployment, especially in resource-constrained environments.