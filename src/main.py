"""
Main script to orchestrate the wildfire detection pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

from pipeline.fetch_data import fetch_and_organize_images
from pipeline.prelabelling.yolo_prelabelling import generate_yolo_prelabelling
from pipeline.prelabelling.sam_prelabelling import generate_segmentation
from pipeline.augmentation import augment_dataset
from pipeline.train import train_model
from pipeline.distill_quantize import distill_model, quantize_model
from pipeline.save_model import register_models
from utils import load_config

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Run the wildfire detection pipeline')
    parser.add_argument(
        '--config',
        type=str,
        help='Path to the pipeline configuration file (default: pipeline_config.json in the same directory as main.py)'
    )
    return parser.parse_args()

def main():
    """
    Main function to orchestrate the entire pipeline.
    """
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    if args.config:
        pipeline_config_path = Path(args.config)
    else:
        pipeline_config_path = SCRIPT_DIR / "pipeline_config.json"
    
    config = load_config(pipeline_config_path)
    
    # Define all paths
    base_dir = Path("mock_io")
    data_dir = base_dir / "data"
    model_dir = base_dir / "model_registry"
    
    # Data paths
    source_dir = data_dir / "sampled_dataset" / "images"
    raw_dir = data_dir / "raw" / "images"
    distilled_dir = data_dir / "raw" / "distilled_images"
    prelabelled_dir = data_dir / "prelabelled"
    processed_dir = data_dir / "processed"
    augmented_dir = data_dir / "augmented"
    training_dir = data_dir / "training"
    distillation_dir = data_dir / "distillation"
    
    # Model paths
    model_path = model_dir / "model" / "nano_trained_model.pt"
    
    # Create necessary directories
    for dir_path in [raw_dir, distilled_dir, prelabelled_dir, processed_dir, 
                    augmented_dir, training_dir, distillation_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(" --- Step 1: Fetching and organizing images --- ")
    # 1. Fetch and organize images
    fetch_and_organize_images(
        source_dir=source_dir,
        raw_dir=raw_dir,
        distilled_dir=distilled_dir,
        config=config,
        seed=config.get('random_seed', 42)
    )
    
    print("-----------------------------------------------\n")
    print(" --- Step 2: Generating YOLO prelabelling --- ")
    
    # 2. Generate predictions for raw images
    generate_yolo_prelabelling(
        raw_dir=raw_dir,
        output_dir=prelabelled_dir / "yolo",
        model_path=model_path,
        config=config
    )
    
    print("-----------------------------------------------\n")
    print(" --- Step 3: Data augmentation --- ")
    
    # 3. Data augmentation
    augment_dataset(
        image_dir=processed_dir,
        output_dir=augmented_dir,
        config=config.get('augmentation_config', {})
    )
    
    print("-----------------------------------------------\n")
    print(" --- Step 4: Model training --- ")
    
    # 4. Model training
    model_path = train_model(
        data_dir=training_dir,
        config=config.get('training_config', {})
    )
    
    print("-----------------------------------------------\n")
    print(" --- Step 5: Model optimization --- ")
    
    # 5. Model optimization
    distilled_model = distill_model(
        model_path=model_path,
        distillation_images=distillation_dir,
        config=config.get('distillation_config', {})
    )
    
    print("-----------------------------------------------\n")
    print(" --- Step 6: Model quantization --- ")
    
    # 6. Model quantization
    quantized_model = quantize_model(
        model_path=distilled_model,
        config=config.get('quantization_config', {})
    )
    
    print("-----------------------------------------------\n")
    print(" --- Step 7: Model registration --- ")
    
    # 7. Model registration
    register_models(
        full_model=model_path,
        distilled_model=distilled_model,
        quantized_model=quantized_model
    )

if __name__ == "__main__":
    main() 