"""
Script to demonstrate YOLO model predictions on raw images.
"""
from pathlib import Path
from typing import List
import json
from src.pipeline.prelabelling.yolo_prelabelling import detect_objects

def process_raw_images() -> None:
    """
    Process all raw images in the mock_io/raw/images directory and save predictions.
    """
    # Define paths
    raw_images_dir = Path("mock_io/raw/images")
    output_dir = Path("mock_io/predictions")
    output_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [
        f for f in raw_images_dir.glob('*')
        if f.suffix.lower() in image_extensions
    ]
    
    # Process each image
    for image_path in image_files:
        try:
            # Get predictions
            predictions = detect_objects(str(image_path))
            
            # Save predictions to JSON file
            output_path = output_dir / f"{image_path.stem}_predictions.json"
            with open(output_path, 'w') as f:
                json.dump(predictions, f, indent=2)
            
            print(f"Processed {image_path.name} -> {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path.name}: {str(e)}")

if __name__ == "__main__":
    process_raw_images() 