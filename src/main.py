"""
Main script to orchestrate the entire wildfire detection pipeline.
"""

import os

from pipeline.fetch_data import fetch_images
from pipeline.labelling import detect_objects, generate_segmentation
from pipeline.augmentation import augment_dataset
from pipeline.train import train_model
from pipeline.distill_quantize import distill_model, quantize_model
from pipeline.save_model import register_models
from utils import load_config


def main():
    # Load configuration
    config = load_config()
    
    # 1. Fetch and organize images
    fetch_images(
        source_path=config.get('source_path', 'data/raw'),
        output_dir=config.get('output_dir', 'data/processed')
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