"""
Main script to orchestrate the wildfire detection pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

from pipeline.fetch_data import fetch_and_organize_images
from pipeline.prelabelling.yolo_prelabelling import generate_yolo_prelabelling
#from pipeline.prelabelling.sam_prelabelling import generate_segmentation
from pipeline.prelabelling.grounding_dino_prelabelling import generate_gd_prelabelling
from pipeline.prelabelling.matching import match_and_filter
from pipeline.augmentation import augment_dataset
from pipeline.train import train_model
from pipeline.distillation import distill_model
from pipeline.quantization import quantize_model
from pipeline.save_model import register_models
from utils import load_config, prepare_training_data

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
    
    # Training configuration
    train_config_path = SCRIPT_DIR / "train_config.json"
    train_config = load_config(train_config_path)

    # Define all paths
    base_dir = Path("mock_io")
    data_dir = base_dir / "data"
    model_dir = base_dir / "model_registry"
    config_dir = base_dir / "config_registry"
    
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
    distilled_output_dir = model_dir / "distilled"
    quantized_output_dir = model_dir / "quantized"
    
    # Create necessary directories
    for dir_path in [raw_dir, distilled_dir, prelabelled_dir, processed_dir, 
                    augmented_dir, training_dir, distillation_dir,
                    distilled_output_dir, quantized_output_dir, config_dir]:
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
    
    # print("-----------------------------------------------\n")
    # print(" --- Step 3: Generating SAM prelabelling --- ")

    # generate_segmentation(
    #     raw_dir=raw_dir,
    #     yolo_json_dir=prelabelled_dir / "yolo",
    #     mask_output_dir=prelabelled_dir / "sam" / "masks",
    #     metadata_output_dir=prelabelled_dir / "sam" / "metadata",
    #     model_path=model_dir / "model" / "mobile_sam.pt",
    #     config=config
    # )
    
    print("-----------------------------------------------\n")
    print(" --- Step 3: Generating Grounding DINO prelabelling --- ")

    
    generate_gd_prelabelling(
        raw_dir=raw_dir,
        output_dir=prelabelled_dir / "gdino",
        config=config,
        model_weights=model_dir / "model" / "groundingdino_swint_ogc.pth",
        config_path=model_dir / "model" / "GroundingDINO_SwinT_OGC.py",
        box_threshold=config.get("dino_box_threshold", 0.3),
        text_threshold=config.get("dino_text_threshold", 0.25)
    )


    print("-----------------------------------------------\n")
    print(" --- Step 4: Matching YOLO and GDINO predictions --- ")

    match_and_filter(
        yolo_dir=prelabelled_dir / "yolo",
        dino_dir=prelabelled_dir / "gdino",
       labeled_dir=Path("mock_io/data/labeled"),
        pending_dir=Path("mock_io/data/mismatched/pending"),
        config=config
    )


    print("-----------------------------------------------\n")
    print(" --- Step 5: Data augmentation --- ")

    # 5. Data augmentation
    augment_dataset(
        image_dir=raw_dir,
        output_dir=augmented_dir,
        config=config.get('augmentation_config', {})
    )

    print("-----------------------------------------------\n")
    print(" --- Step 6: Model training --- ")

    # 6. Model training
    prepare_training_data(config)
    model_path = train_model(train_config)

    print("-----------------------------------------------\n")
    print(" --- Step 7: Model optimization --- ")

    # 7. Model optimization
    distilled_model_path, distill_config_path = distill_model(
        model_path=model_path,
        distillation_images=distillation_dir,
        config=config,
        output_dir=distilled_output_dir,
       config_registry_path=config_dir
    )

    print("-----------------------------------------------\n")
    print(" --- Step 8: Model quantization --- ")
    # 8. Model quantization
    # Replace with distilled_model, this is for testing using the full model
    distilled_model_path = model_dir / "model" / "nano_trained_model.pt"
    quantized_model = quantize_model(
        model_path=distilled_model_path,
        config={
            'method': config.get('quantization_method'),
            'output_dir': str(quantized_output_dir)
        }
    )

    print("-----------------------------------------------\n")
    print(" --- Step 9: Model registration --- ")

    # 9. Model registration
    register_models(
        full_model=model_path,
        distilled_model=distilled_model_path,
        quantized_model=quantized_model_path
    )

if __name__ == "__main__":
    main()
