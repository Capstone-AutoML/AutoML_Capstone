"""
Main script to orchestrate the wildfire detection pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

from ultralytics.utils import YAML

from pipeline import (
    validate_input_images,
    generate_yolo_prelabelling,
    generate_gd_prelabelling,
    match_and_filter,
    run_human_review,
    augment_dataset,
    train_model,
    start_distillation,
    quantize_model,
    register_models
)
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
        help='Path to the pipeline configuration file (default: pipeline_config.json in config directory)'
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

    # Define all paths
    workspace_dir = PROJECT_ROOT / "automl_workspace"
    config_dir = workspace_dir / "config"
    data_pipeline_dir = workspace_dir / "data_pipeline"
    master_dataset_dir = workspace_dir / "master_dataset"
    model_registry_dir = workspace_dir / "model_registry"

    # Load configurations
    if args.config:
        pipeline_config_path = Path(args.config)
    else:
        pipeline_config_path = config_dir / "pipeline_config.json"

    config = load_config(pipeline_config_path)

    # Get process options from config
    process_options = config.get("process_options", {})
    skip_human_review = process_options.get("skip_human_review", False)
    skip_training = process_options.get("skip_training", False)
    skip_distillation = process_options.get("skip_distillation", False)
    skip_quantization = process_options.get("skip_quantization", False)

    print(" --- PIPELINE CONFIGURATION --- ")
    print(f"Human Review: {'Disabled' if skip_human_review else 'Enabled'}")
    print(f"Training: {'Disabled' if skip_training else 'Enabled'}")
    print(f"Distillation: {'Disabled' if skip_distillation else 'Enabled'}")
    print(f"Quantization: {'Disabled' if skip_quantization else 'Enabled'}")
    print("-----------------------------------------------\n")

    # Training configuration
    train_config_path = config_dir / "train_config.json"
    train_config = load_config(train_config_path)

    # Distillation configuration
    # distillation_config_path = SCRIPT_DIR / "distillation_config.json"
    # distillation_config = load_config(distillation_config_path)
    distillation_config_path = config_dir / "distillation_config.yaml"
    distillation_config = YAML.load(distillation_config_path)

    # Quantization configuration
    quantize_config_path = config_dir / "quantize_config.json"

    # Data paths
    source_dir = data_pipeline_dir / "input"
    prelabelled_dir = data_pipeline_dir / "prelabeled"
    labeled_dir = data_pipeline_dir / "labeled"
    augmented_dir = data_pipeline_dir / "augmented"
    training_dir = data_pipeline_dir / "training"
    distillation_dir = data_pipeline_dir / "distillation"
    quantization_dir = data_pipeline_dir / "quantization"

    # Model paths
    base_model_path = model_registry_dir / "model" / "nano_trained_model.pt"
    trained_model_path = None
    distilled_model_path = None
    quantized_model_path = None
    distilled_output_dir = model_registry_dir / "distilled"
    quantized_output_dir = model_registry_dir / "quantized"

    # Label Studio data paths
    label_studio_dir = data_pipeline_dir / "label_studio"
    pending_dir = label_studio_dir / "pending"
    tasks_dir = label_studio_dir / "tasks"
    results_dir = label_studio_dir / "results"

    # Track the current model in pipeline
    current_model_path = base_model_path

    print(" --- Step 1: Validating images in input folder --- ")
    # 1. Validate images in input folder
    try:
        validate_input_images(input_dir=source_dir)
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    print("-----------------------------------------------\n")
    print(" --- Step 2: Generating YOLO prelabelling --- ")

    # 2. Generate predictions for raw images
    generate_yolo_prelabelling(
        raw_dir=source_dir,
        output_dir=prelabelled_dir / "yolo",
        model_path=base_model_path,
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
    print(" --- Step 5: Human intervention --- ")

    # 5. Human intervention
    if skip_human_review:
        print("[Info] Human review is disabled, skipping...")
    else:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        if not api_key:
            print("Please set LABEL_STUDIO_API_KEY in the .env file")
            exit(1)

        review_results = run_human_review(
            project_name="AutoML-Human-Intervention",
            export_results_flag=None
        )
        if not review_results:
            print("[Error] Human review process failed")
            sys.exit(1)

        print(f"[✓] Human review completed with {len(review_results)} reviewed items")

    print("-----------------------------------------------\n")
    print(" --- Step 6: Data augmentation --- ")

    # 6. Data augmentation
    augment_dataset(
        image_dir=source_dir,
        output_dir=augmented_dir,
        config=config.get('augmentation_config', {})
    )

    print("-----------------------------------------------\n")
    print(" --- Step 7: Model training --- ")

    # 7. Model training
    if skip_training:
        print("[Info] Training is disabled, skipping...")
    else:
        prepare_training_data(config)
        trained_model_path = train_model(train_config)
        current_model_path = trained_model_path

    print("-----------------------------------------------\n")
    print(" --- Step 8: Model Distillation --- ")

    # 8. Model Distillation
    if skip_distillation:
        print("[Info] Distillation is disabled, skipping...")
    else:
        # Define distillation hyperparameters
        distillation_hyperparams = {
            "lambda_distillation": 2.0,
            "lambda_detection": 1.0,
            "lambda_dist_ciou": 1.0,
            "lambda_dist_kl": 2.0,
            "temperature": 2.0
        }

        # Start distillation process using current model
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
        current_model_path = distilled_model_path

    print("-----------------------------------------------\n")
    print(" --- Step 9: Model quantization --- ")
    # 9. Model quantization
    if skip_quantization:
        print("[Info] Quantization is disabled, skipping...")
    else:
        quantized_model_path = quantize_model(
            model_path=str(current_model_path),
            quantize_config_path=str(quantize_config_path)
        )

    print("-----------------------------------------------\n")
    print(" --- Step 10: Model registration --- ")

    # 10. Model registration
    print("[Info] Registering models:")
    print(f"Base model: {base_model_path}")
    if trained_model_path:
        print(f"Trained model: {trained_model_path}")
    if distilled_model_path:
        print(f"Distilled model: {distilled_model_path}")
    if quantized_model_path:
        print(f"Quantized model: {quantized_model_path}")

    register_models(
        full_model=trained_model_path,
        distilled_model=distilled_model_path,
        quantized_model=quantized_model_path
    )


if __name__ == "__main__":
    main()
