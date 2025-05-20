"""
Script for model quantization to reduce size and improve inference speed.
"""
from pathlib import Path
from typing import Dict
from ultralytics import YOLO

def quantize_model(
    model_path: str,
    config: dict,
    output_dir: Path
) -> str:
    """
    Apply quantization to the model to reduce size and improve inference speed.
    
    Args:
        model_path (str): Path to the model to quantize
        config (dict): Quantization configuration parameters
        output_dir (Path): Directory to save the quantized model
        
    Returns:
        str: Path to the quantized model
    """
    # Load the model
    model = YOLO(model_path)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get quantization parameters from config or use defaults
    quant_config = config.get('quantization_config', {})
    format_type = quant_config.get('format', 'onnx')
    int8 = quant_config.get('int8', True)
    
    # Export the model with quantization
    quantized_model_path = model.export(
        format=format_type,
        int8=int8,
        project=str(output_dir),
        name='quantized_model'
    )
    
    return quantized_model_path 