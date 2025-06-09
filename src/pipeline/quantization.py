"""
Script for model quantization to reduce size and improve inference speed.
"""

import os
import torch
import shutil
import platform
from ultralytics import YOLO
import subprocess
from datetime import datetime
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
from utils import prepare_quantization_data, load_config


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

    try:
        # Export the model to IMX format (no fraction parameter)
        export_result = model.export(format="imx", data=quantize_yaml, device=device)
        if isinstance(export_result, (list, tuple)) and len(export_result) > 0:
            exported_path = export_result[0]
        else:
            exported_path = str(export_result)

        if os.path.exists(exported_path):
            if os.path.exists(output_path):
                if os.path.isdir(output_path):
                    shutil.rmtree(output_path)
                else:
                    os.remove(output_path)
            shutil.move(exported_path, output_path)
            print(f"[INFO] IMX quantized model saved to {output_path}")
            return output_path
        else:
            print(f"[ERROR] Export result not found: {exported_path}")
            return None

    except Exception as e:
        print(f"[ERROR] IMX quantization failed: {str(e)}")
        return None


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
    try:
        model.model = model.model.half()
        model.save(output_path)
        print(f"[INFO] FP16 quantized model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"[ERROR] FP16 quantization failed: {str(e)}")
        return None


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
    try:
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

    except Exception as e:
        print(f"[ERROR] ONNX quantization failed: {str(e)}")
        return None


def quantize_model(model_path: str, quantize_config_path: str) -> str:
    """
    Apply quantization to the model to reduce size and improve inference speed.

    Args:
        model_path (str): Path to the model to quantize
        quantize_config_path (str): Path to quantization configuration file

    Returns:
        str: Path to the quantized model
    """
    print(f"[INFO] Loading quantization config from {quantize_config_path}")
    quantize_config = load_config(quantize_config_path)

    output_dir = quantize_config.get('output_dir', 'quantized_models')
    quant_method = quantize_config.get('quantization_method', 'IMX')
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = YOLO(model_path)

    # Generate base paths
    model_name = os.path.basename(str(model_path))
    base_name = os.path.splitext(model_name)[0]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

    quantize_yaml = None

    # Prepare calibration data if using IMX quantization
    if quant_method == "IMX":
        print("[INFO] Preparing calibration data for IMX quantization...")

        labeled_images_dir = Path(quantize_config["labeled_images_path"])
        calibration_samples = quantize_config.get("calibration_samples", 200)

        prepare_quantization_data(quantize_config, labeled_images_dir, calibration_samples)

        quantize_yaml = quantize_config.get("quantize_yaml_path", "src/quantize.yaml")

        if not os.path.exists(quantize_yaml):
            raise ValueError(f"The quantize_yaml file does not exist at: {quantize_yaml}")

    # Quantize the model based on the configured method
    if quant_method == "ONNX":
        preprocessed_path = os.path.join(output_dir, f"{timestamp}_{base_name}_preprocessed.onnx")
        output_path = os.path.join(output_dir, f"{timestamp}_{base_name}_onnx_quantized_model.onnx")
        return onnx_quantization(model, output_path, preprocessed_path)

    elif quant_method == "FP16":
        output_path = os.path.join(output_dir, f"{timestamp}_{base_name}_fp16_quantized_model.pt")
        return fp16_quantization(model, output_path)

    elif quant_method == "IMX":
        output_path = os.path.join(output_dir, f"{timestamp}_{base_name}_imx_quantized_model")
        return imx_quantization(model, output_path, quantize_yaml)

    else:
        raise ValueError(f"Unsupported quantization method: {quant_method}")
