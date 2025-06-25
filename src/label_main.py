"""
Labeling main script to orchestrate the wildfire detection pipeline.
"""

import argparse
from pathlib import Path

from pipeline import (
    validate_input_images,
    generate_yolo_prelabelling,
    generate_gd_prelabelling,
    match_and_filter
)
from directory_setup import create_automl_workspace
from utils import load_config

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
    Main function to orchestrate the labeling pipeline.
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

    print(" --- Step 1: Validating images in input folder --- ")
    # 1. Validate images in input folder
    try:
        validate_input_images(input_dir=pipeline_paths["source_dir"])
    except ValueError as e:
        print(f"[ERROR] {e}")
        return

    print("-----------------------------------------------\n")
    print(" --- Step 2: Generating YOLO prelabelling --- ")

    # 2. Generate predictions for raw images
    generate_yolo_prelabelling(
        raw_dir=pipeline_paths["source_dir"],
        output_dir=pipeline_paths["prelabeled_dir"] / "yolo",
        model_path=pipeline_paths["base_model_path"],
        config=config
    )

    print("-----------------------------------------------\n")
    print(" --- Step 3: Generating Grounding DINO prelabelling --- ")

    generate_gd_prelabelling(
        raw_dir=pipeline_paths["source_dir"],
        output_dir=pipeline_paths["prelabeled_dir"] / "gdino",
        config=config,
        model_weights=pipeline_paths["grounding_dino_weights"],
        config_path=pipeline_paths["grounding_dino_config"],
        box_threshold=config.get("dino_box_threshold", 0.3),
        text_threshold=config.get("dino_text_threshold", 0.25)
    )

    print("-----------------------------------------------\n")
    print(" --- Step 4: Matching YOLO and GDINO predictions --- ")

    match_and_filter(
        yolo_dir=pipeline_paths["prelabeled_dir"] / "yolo",
        dino_dir=pipeline_paths["prelabeled_dir"] / "gdino",
        labeled_dir=pipeline_paths["labeled_dir"],
        pending_dir=pipeline_paths["pending_dir"],
        config=config
    )

    
if __name__ == "__main__":
    main()
