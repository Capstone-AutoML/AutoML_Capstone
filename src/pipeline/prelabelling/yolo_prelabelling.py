"""
Script for automated pre-labelling using YOLO and SAM models.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import torch
from ultralytics import YOLO
import json
import sys
import os
from tqdm import tqdm

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _detect_device() -> str:
    """
    Detect and return the best available device for model inference.
    
    Returns:
        str: Device name ('cuda', 'mps', or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def _load_model(model_path: Path, device: str) -> YOLO:
    """
    Load YOLO model and move it to the specified device.
    
    Args:
        model_path (Path): Path to the YOLO model file
        device (str): Device to load the model on
        
    Returns:
        YOLO: Loaded YOLO model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = YOLO(str(model_path))
    model.to(device)
    return model

def _get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files from the specified directory.
    
    Args:
        directory (Path): Directory to search for images
        
    Returns:
        List[Path]: List of paths to image files
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    return [
        f for f in directory.glob('*')
        if f.suffix.lower() in image_extensions
    ]

def _process_prediction(result) -> List[Dict[str, Union[float, str, List[float]]]]:
    """
    Process a single prediction result into a standardized format.
    
    Args:
        result: YOLO prediction result
        
    Returns:
        List[Dict[str, Union[float, str, List[float]]]]: List of processed predictions
    """
    predictions = []
    for box in result.boxes:
        # Get coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Get confidence score
        confidence = float(box.conf[0])
        
        # Get class name
        class_id = int(box.cls[0])
        class_name = result.names[class_id]
        
        predictions.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': confidence,
            'class': class_name
        })
    return predictions

def _save_predictions(predictions: List[Dict], output_path: Path) -> None:
    """
    Save predictions to a JSON file.
    
    Args:
        predictions (List[Dict]): List of predictions to save
        output_path (Path): Path to save the predictions
    """
    with open(output_path, 'w') as f:
        json.dump({'predictions': predictions}, f, indent=2)

def generate_yolo_prelabelling(raw_dir: Path, output_dir: Path, model_path: Path, config: Dict, verbose: bool = False) -> None:
    """
    Generate predictions for all images in the raw directory using YOLO model.
    
    Args:
        raw_dir (Path): Path to directory containing raw images
        output_dir (Path): Path to save prediction results
        model_path (Path): Path to the YOLO model file
        config (Dict): Configuration dictionary containing pipeline parameters
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect and set device
    device = config.get("torch_device", "auto")
    if device == "auto":
        device = _detect_device()
    print(f"Using device: {device}")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Load YOLO model
    model = _load_model(model_path, device)
    print(f"Loaded YOLO model from {model_path}")
    
    # Get all image files
    image_files = _get_image_files(raw_dir)
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    failed = 0
    
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Run inference and process results
            results = model(str(image_path), verbose=verbose)
            predictions = []
            for result in results:
                predictions.extend(_process_prediction(result))
            
            # Save predictions
            output_path = output_dir / f"{image_path.stem}.json"
            _save_predictions(predictions, output_path)
            
            successful += 1
            if verbose:
                print(f"Processed {image_path.name} -> {output_path}")
            
        except Exception as e:
            failed += 1
            print(f"Error processing {image_path.name}: {str(e)}")
    
    print(f"\nPrediction Summary:")
    print(f"Successfully processed: {successful} images")
    print(f"Failed to process: {failed} images")
