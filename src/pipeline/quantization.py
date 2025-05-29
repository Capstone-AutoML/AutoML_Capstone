"""
Script for model quantization to reduce size and improve inference speed.
"""

import os
import torch
import shutil
import platform
from ultralytics import YOLO
import subprocess
from onnxruntime.quantization import quantize_dynamic, QuantType


def imx_quantization(model, output_path, quantize_yaml):
    """
    Apply IMX post training quantization to the model.

    Args:
        model: YOLO model to quantize
        output_path: Path to save the quantized model
        quantize_yaml: Path to quantize.yaml with nc and class names

    Returns:
        str or None: Path to the quantized model, or None if on non-Linux platforms.
    """
    # Check if the platform is Linux
    if platform.system().lower() != "linux":
        print("❌ IMX quantization export is only supported on Linux. Skipping...")
        return None

    device = 0 if torch.cuda.is_available() else "cpu"
    # Apply IMX quantization
    print(f"Applying IMX quantization using device {device}...")

    # Export the model to IMX format
    export_result = model.export(format="imx", data=quantize_yaml, device=device)
    exported_path = export_result[0]

    shutil.move(exported_path, output_path)
    print(f"IMX quantized model saved to {output_path}")
    return output_path


def fp16_quantization(model, output_path):
    """
    Apply FP16 quantization to the model.

    Args:
        model: YOLO model to quantize
        output_path: Path to save the quantized model

    Returns:
        str: Path to the quantized model
    """
    # Apply FP16 quantization
    print("Applying FP16 quantization...")
    model.model = model.model.half()

    # Save the quantized model
    model.save(output_path)

    print(f"FP16 quantized model saved to {output_path}")
    return output_path


def onnx_quantization(model, output_path, preprocessed_path):
    """
    Apply ONNX dynamic quantization to the model.

    Args:
        model: YOLO model to quantize
        output_path: Path to save the final quantized model
        preprocessed_path: Path to save the preprocessed ONNX model

    Returns:
        str: Path to the quantized model
    """
    # Export the model to ONNX format
    print("Exporting model to ONNX format...")
    onnx_path = model.export(format='onnx')

    # Preprocess the ONNX model
    print("Preprocessing ONNX model...")
    subprocess.run([
        "python", "-m", "onnxruntime.quantization.preprocess",
        "--input", onnx_path,
        "--output", preprocessed_path
    ], check=True)

    # Apply dynamic quantization
    print("Applying dynamic quantization...")
    quantize_dynamic(
        preprocessed_path,
        output_path,
        weight_type=QuantType.QUInt8
    )

    # Clean up the preprocessed model
    if os.path.exists(onnx_path) and onnx_path != output_path:
        os.remove(onnx_path)

    print(f"ONNX quantized model saved to {output_path}")
    return output_path


def quantize_model(model_path: str, config: dict, quantize_yaml: str = None) -> str:
    """
    Apply quantization to the model to reduce size and improve inference speed.

    Args:
        model_path (str): Path to the model to quantize
        config (dict): Quantization configuration parameters
        quantize_yaml (str): Path to YAML file for IMX export

    Returns:
        str: Path to the quantized model
    """
    output_dir = config.get('output_dir', 'quantized_models')
    quant_method = config.get('method')
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = YOLO(model_path)

    # Generate base paths
    model_name = os.path.basename(str(model_path))
    base_name = os.path.splitext(model_name)[0]

    # Quantize the model based on the configured method
    if quant_method == "ONNX":
        preprocessed_path = os.path.join(output_dir, f"{base_name}_preprocessed.onnx")
        output_path = os.path.join(output_dir, f"{base_name}_onnx_quantized.onnx")
        return onnx_quantization(model, output_path, preprocessed_path)

    elif quant_method == "FP16":
        output_path = os.path.join(output_dir, f"{base_name}_fp16.pt")
        return fp16_quantization(model, output_path)

    elif quant_method == "IMX":
        if not quantize_yaml:
            raise ValueError("IMX quantization requires a quantize_yaml path.")
        if not os.path.exists(quantize_yaml):
            raise ValueError(f"The quantize_yaml file does not exist at the specified path: {quantize_yaml}")
        output_path = os.path.join(output_dir, f"{base_name}_imx_quantized.pt")
        return imx_quantization(model, output_path, quantize_yaml)

    else:
        raise ValueError(f"Unsupported quantization method: {quant_method}")
