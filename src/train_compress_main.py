"""
Training to compression main script to orchestrate the wildfire detection pipeline.
"""

import argparse
from pathlib import Path
from ultralytics.utils import YAML

from pipeline import (
    augment_dataset,
    train_model,
    start_distillation,
    quantize_model,
    register_models,
    clean_pipeline_workspace
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
    Main function to orchestrate training and compression pipeline.
    """
    # Parse command line arguments
    args = parse_args()

    # Create the directory structure
    create_automl_workspace(base_path=PROJECT_ROOT)

    # Load configurations
    if args.config:
        pipeline_config_path = Path(args.config)
    else:
        pipeline_config_path = PROJECT_ROOT / "automl_workspace" / "config" / "pipeline_config.json"

    config = load_config(pipeline_config_path)

    # Convert relative paths from config to absolute paths
    pipeline_paths = {}
    for path_key, relative_path in config.get("pipeline_paths", {}).items():
        pipeline_paths[path_key] = PROJECT_ROOT / relative_path

    # Get process options from config
    process_options = config.get("process_options", {})
    skip_training = process_options.get("skip_training", False)
    skip_distillation = process_options.get("skip_distillation", False)
    skip_quantization = process_options.get("skip_quantization", False)


    # Augmentation configuration
    augmentation_config = load_config(pipeline_paths["augmentation_config_path"])

    # Training configuration
    train_config = load_config(pipeline_paths["train_config_path"])

    # Distillation configuration
    distillation_config = YAML.load(pipeline_paths["distillation_config_path"])

    # Quantization configuration path
    quantize_config_path = pipeline_paths["quantize_config_path"]

    # Initialize model path variables
    trained_model_path = None
    distilled_model_path = None
    quantized_model_path = None

    # Track the current model in pipeline
    current_model_path = pipeline_paths["base_model_path"]

   
    print(" --- Step 6: Data augmentation --- ")

    # 6. Data augmentation
    augment_dataset(
        image_dir=pipeline_paths["source_dir"],
        output_dir=pipeline_paths["augmented_dir"],
        config=augmentation_config
    )

    print("-----------------------------------------------\n")
    print(" --- Step 7: Model training --- ")

    # 7. Model training
    if skip_training:
        print("[Info] Training is disabled, skipping...")
    else:
        prepare_training_data(train_config)
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
            img_dir=pipeline_paths["distillation_dir"] / "distillation_dataset",
            frozen_layers=10,
            save_checkpoint_every=25,
            hyperparams=distillation_hyperparams,
            resume_checkpoint=None,
            output_dir=pipeline_paths["distilled_output_dir"],
            final_model_dir=pipeline_paths["distilled_output_dir"] / "latest",
            log_level="batch",
            debug=False,
            distillation_config=distillation_config,
            pipeline_config=config
        )

        # Get the path to the distilled model
        distilled_model_path = pipeline_paths["distilled_output_dir"] / "latest" / "model.pt"
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
    print(f"Base model: {pipeline_paths['base_model_path']}")
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

    print("-----------------------------------------------\n")
    print(" --- Step 11: Final Cleanup and Archival --- ")
    clean_pipeline_workspace(
        data_pipeline_dir=pipeline_paths["data_pipeline_dir"],
        master_dataset_dir=pipeline_paths["master_dataset_dir"]
    )


if __name__ == "__main__":
    main()
