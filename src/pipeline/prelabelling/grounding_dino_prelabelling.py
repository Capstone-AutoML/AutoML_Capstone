import os
import json
import warnings
import cv2
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import contextlib
import io

# Suppress warnings for clean output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Default detection classes and thresholds
TEXT_PROMPTS = ["fire", "smoke", "person", "vehicle", "lightning"]
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25


def _get_image_files(directory: Path) -> List[Path]:
    """
    Get all image files (jpg, jpeg, png) from the given directory.
    """
    return [f for f in directory.glob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]


def generate_gd_prelabelling(
    raw_dir: Path,
    output_dir: Path,
    config: Dict,
    model_weights: Path,
    config_path: Path,
    text_prompts: List[str] = TEXT_PROMPTS,
    box_threshold: float = None,
    text_threshold: float = None
) -> None:
    """
    Run Grounding DINO to detect objects from images using text prompts.
    Saves a JSON file per image with predicted bounding boxes and metadata.
    This version runs sequentially (no multiprocessing).
    """
    print(f"Using device: {config.get('torch_device', 'cpu')}")

    # Get thresholds from config or fallback to defaults
    box_threshold = config.get("dino_box_threshold", BOX_THRESHOLD)
    text_threshold = config.get("dino_text_threshold", TEXT_THRESHOLD)

    # Get list of all images to process
    image_files = _get_image_files(raw_dir)
    print(f"Found {len(image_files)} images to process")

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Grounding DINO model here to keep top-level clean
    from groundingdino.util.inference import Model

    # Load the model once
    with contextlib.redirect_stdout(io.StringIO()):  # Suppress internal model logs
        model = Model(
            model_config_path=str(config_path),
            model_checkpoint_path=str(model_weights),
            device=config.get("torch_device", "cpu")
        )

    # Track results for summary
    success_count = 0
    skipped_count = 0
    error_count = 0

    # Process each image one by one
    for image_path in tqdm(image_files, desc="Processing images"):
        image_name = image_path.stem
        image = cv2.imread(str(image_path))

        if image is None:
            skipped_count += 1
            continue

        try:
            # Run inference with class prompts
            detections = model.predict_with_classes(
                image=image,
                classes=text_prompts,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )

            # Format output
            metadata = []
            for box, score, class_id in zip(detections.xyxy, detections.confidence, detections.class_id):
                label = text_prompts[class_id] if class_id is not None else "unknown"
                metadata.append({
                    "image": image_name,
                    "class": label,
                    "confidence": float(score),
                    "bbox": [float(x) for x in box],
                    "source": "grounding_dino"
                })

            # Save results to JSON
            output_file = output_dir / f"{image_name}.json"
            with open(output_file, "w") as f:
                json.dump({"predictions": metadata}, f, indent=2)

            success_count += 1

        except Exception as e:
            error_count += 1
            continue

    # Print summary
    print("\nPrediction Summary:")
    print(f"Successfully processed: {success_count} images")
    print(f"Skipped (unreadable): {skipped_count} images")
    print(f"Failed to process: {error_count} images")
