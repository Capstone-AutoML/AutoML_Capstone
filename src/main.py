"""
Main script to orchestrate the wildfire detection pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

from pipeline.fetch_data import fetch_and_organize_images
from pipeline.labelling import detect_objects, generate_segmentation
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
    
    # Define paths
    base_dir = Path("mock_io/data")
    source_dir = base_dir / "sampled_dataset" / "images"
    raw_dir = base_dir / "raw" / "images"
    distilled_dir = base_dir / "raw" / "distilled_images"
    
    # Fetch and organize images
    fetch_and_organize_images(
        source_dir=source_dir,
        raw_dir=raw_dir,
        distilled_dir=distilled_dir,
        config=config,
        seed=config.get('random_seed', 42)
    )
    
    # 2. Pre-labelling with YOLO and SAM
    # TODO: Implement batch processing of images
    image_path = "path/to/image.jpg"
    bounding_boxes = detect_objects(image_path)
    segmentation_masks = generate_segmentation(image_path, bounding_boxes)
    
    # 3. Data augmentation
    augment_dataset(
        image_dir=config.get('image_dir', 'data/processed'),
        output_dir=config.get('augmented_dir', 'data/augmented'),
        config=config.get('augmentation_config', {})
    )
    
    # 4. Model training
    model_path = train_model(
        data_dir=config.get('training_dir', 'data/augmented'),
        config=config.get('training_config', {})
    )
    
    # 5. Model optimization
    distilled_model = distill_model(
        model_path=model_path,
        distillation_images=config.get('distillation_images', 'data/distillation'),
        config=config.get('distillation_config', {})
    )
    
    quantized_model = quantize_model(
        model_path=distilled_model,
        config=config.get('quantization_config', {})
    )
    
    # 6. Model registration
    register_models(
        full_model=model_path,
        distilled_model=distilled_model,
        quantized_model=quantized_model
    )

if __name__ == "__main__":
    main() 