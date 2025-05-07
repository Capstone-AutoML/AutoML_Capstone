"""
Main script to orchestrate the wildfire detection pipeline.
"""

from pathlib import Path
from pipeline.fetch_data import fetch_and_organize_images
from pipeline.labelling import detect_objects, generate_segmentation
from pipeline.augmentation import augment_dataset
from pipeline.train import train_model
from pipeline.distill_quantize import distill_model, quantize_model
from pipeline.save_model import register_models
from utils import load_config


def main():
    """
    Main function to orchestrate the entire pipeline.
    """
    # Load configuration
    config = load_config()
    
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
        sample_size=config.get('sample_size', 100),
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