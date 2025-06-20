"""
Main script to orchestrate the wildfire detection pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

from ultralytics.utils import YAML

from pipeline.fetch_data import fetch_and_organize_images
from pipeline.prelabelling.yolo_prelabelling import generate_yolo_prelabelling
from pipeline.prelabelling.grounding_dino_prelabelling import generate_gd_prelabelling
from pipeline.prelabelling.matching import match_and_filter
from pipeline.augmentation import augment_dataset
from pipeline.train import train_model
from pipeline.distillation.distillation import start_distillation
from pipeline.quantization import quantize_model
from pipeline.save_model import register_models
from directory_setup import create_automl_workspace
from utils import load_config, prepare_training_data, detect_device

# Get the directory containing this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

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

    # Create the directory structure
    create_automl_workspace(base_path=PROJECT_ROOT)

    # Load configurations
    if args.config:
        pipeline_config_path = Path(args.config)
    else:
        pipeline_config_path = SCRIPT_DIR / "pipeline_config.json"
    
    config = load_config(pipeline_config_path)
    
    # Training configuration
    train_config_path = SCRIPT_DIR / "train_config.json"
    train_config = load_config(train_config_path)

    # Distillation configuration
    # distillation_config_path = SCRIPT_DIR / "distillation_config.json"
    # distillation_config = load_config(distillation_config_path)
    distillation_config_path = SCRIPT_DIR / "distillation_config.yaml"
    distillation_config = YAML.load(distillation_config_path)

    # Define all paths
    workspace_dir = PROJECT_ROOT / "automl_workspace"
    config_dir = workspace_dir / "config"
    data_pipeline_dir = workspace_dir / "data_pipeline"
    master_dataset_dir = workspace_dir / "master_dataset"
    model_registry_dir = workspace_dir / "model_registry"

    # Data paths
    source_dir = data_pipeline_dir / "input"
    prelabelled_dir = data_pipeline_dir / "prelabeled"
    labeled_dir = data_pipeline_dir / "labeled"
    augmented_dir = data_pipeline_dir / "augmented"
    training_dir = data_pipeline_dir / "training"
    distillation_dir = data_pipeline_dir / "distillation"
    quantization_dir = data_pipeline_dir / "quantization"

    # Model paths
    model_path = model_registry_dir / "model" / "nano_trained_model.pt"
    distilled_output_dir = model_registry_dir / "distilled"
    quantized_output_dir = model_registry_dir / "quantized"

    # Label Studio data paths
    label_studio_dir = data_pipeline_dir / "label_studio"
    pending_dir = label_studio_dir / "pending"
    tasks_dir = label_studio_dir / "tasks"
    results_dir = label_studio_dir / "results"

    print(" --- Step 1: Fetching images from input folder --- ")
    # 1. Fetch images from input folder
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    images = [f for f in source_dir.glob('*') if f.is_file() and f.suffix.lower() in image_extensions]
    print(f"Found {len(images)} images in {source_dir}")

    if len(images) == 0:
        print("[ERROR] No images found in input directory.\
            Please add images to automl_workspace/data_pipeline/input/")
        return

    print("-----------------------------------------------\n")
    print(" --- Step 2: Generating YOLO prelabelling --- ")
    
    # 2. Generate predictions for raw images
    generate_yolo_prelabelling(
        raw_dir=source_dir,
        output_dir=prelabelled_dir / "yolo",
        model_path=model_path,
        config=config
    )
    
    print("-----------------------------------------------\n")
    print(" --- Step 3: Generating Grounding DINO prelabelling --- ")

    generate_gd_prelabelling(
        raw_dir=source_dir,
        output_dir=prelabelled_dir / "gdino",
        config=config,
        model_weights=model_registry_dir / "model" / "groundingdino_swinb_cogcoor.pth",
        config_path=model_registry_dir / "model" / "GroundingDINO_SwinB_cfg.py",
        box_threshold=config.get("dino_box_threshold", 0.3),
        text_threshold=config.get("dino_text_threshold", 0.25)
    )

    print("-----------------------------------------------\n")
    print(" --- Step 4: Matching YOLO and GDINO predictions --- ")

    match_and_filter(
        yolo_dir=prelabelled_dir / "yolo",
        dino_dir=prelabelled_dir / "gdino",
        labeled_dir=labeled_dir,
        pending_dir=pending_dir,
        config=config
    )

    print("-----------------------------------------------\n")
    print(" --- Step 5: Data augmentation --- ")

    # 5. Data augmentation
    augment_dataset(
        image_dir=source_dir,
        output_dir=augmented_dir,
        config=config.get('augmentation_config', {})
    )

    print("-----------------------------------------------\n")
    print(" --- Step 6: Model training --- ")

    # 6. Model training
    prepare_training_data(config)
    model_path = train_model(train_config)

    print("-----------------------------------------------\n")
    print(" --- Step 7: Model Distillation --- ")

    # 7. Model Distillation
    # Define distillation hyperparameters
    distillation_hyperparams = {
        "lambda_distillation": 2.0,
        "lambda_detection": 1.0,
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0,
        "temperature": 2.0
    }
    
    # Start distillation process
    start_distillation(
        device=config.get("torch_device", "cpu") if config.get("torch_device", "cpu") else detect_device(),
        base_dir=SCRIPT_DIR,
        img_dir=distillation_dir / "distillation_dataset",
        frozen_layers=10,  # Freeze backbone layers
        save_checkpoint_every=25,
        hyperparams=distillation_hyperparams,
        resume_checkpoint=None,  # Can be set to resume from a checkpoint if needed
        output_dir=distilled_output_dir,
        final_model_dir=distilled_output_dir / "latest",
        log_level="batch",
        debug=False,
        distillation_config=distillation_config,
        pipeline_config=config
    )
    
    # Get the path to the distilled model
    distilled_model_path = distilled_output_dir / "latest" / "model.pt"

    print("-----------------------------------------------\n")
    print(" --- Step 8: Model quantization --- ")
    # 8. Model quantization
    quantize_config_path = SCRIPT_DIR / "quantize_config.json"
    quantized_model_path = quantize_model(
        model_path=str(distilled_model_path),
        quantize_config_path=str(quantize_config_path)
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
