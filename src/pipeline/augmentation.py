# import libraries
import cv2
import json
from pathlib import Path
from PIL import Image
import albumentations as A

# Create augmentation pipeline
def build_augmentation_transform(config: dict) -> A.Compose:
    """Build the augmentation transform pipeline from config."""
    return A.Compose([
        # Flips image horizontally, applied 50% of the time by default
        A.HorizontalFlip(p=config.get("horizontal_flip_prob", 0.5)),
        # Alters brightness and contrast, applied 50% of the time by default 
        A.RandomBrightnessContrast(p=config.get("brightness_contrast_prob", 0.5)),
        # Distorts colors in image
        A.HueSaturationValue(p=config.get("hue_saturation_prob", 0.5)),
        # Blurs image, kernel size (blur_limit=3) is 3x3
        A.Blur(blur_limit=3, p=config.get("blur_prob", 0.3)),
        # Adds noise to image, intensity ranges from 10-50
        A.GaussNoise(var_limit=(10.0, 50.0), p=config.get("gauss_noise_prob", 0.3)),
        # Converts image to grayscale
        A.ToGray(p=config.get("grayscale_prob", 0.2)),
        # Rotates image up to 15 degrees, fille empty borders with black
        A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=config.get("rotate_prob", 0.4)),
    # Transform bounding boxes
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def augment_images(matched_pairs: list, transform: A.Compose, output_img_dir: Path, output_json_dir: Path, num_augmentations: int) -> None:
    """Applies augmentations to each image N times and saves results."""

    # Create folders if they don't exist
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)

    # Read images and json label file
    for json_path, image_path in matched_pairs:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(json_path, "r") as f:
            data = json.load(f)

        bboxes = [obj["bbox"] for obj in data["predictions"]]
        class_labels = [obj["class"] for obj in data["predictions"]]
        confidences = [obj["confidence"] for obj in data["predictions"]]

        # Skip images with no bounding box
        if not bboxes:
            continue

        # Apply augmentation and update bounding box
        for i in range(num_augmentations):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

            # Save augmented image
            aug_id = f"{image_path.stem}_aug{i+1}"
            aug_image_path = output_img_dir / f"{aug_id}.jpg"
            Image.fromarray(aug_image).save(aug_image_path)

            aug_predictions = []
            for box, cls, conf in zip(aug_bboxes, aug_classes, confidences):
                aug_predictions.append({
                    "bbox": [round(x, 2) for x in box],
                    "confidence": round(conf, 3),
                    "class": cls
                })

            # Save json labels
            aug_json = {"predictions": aug_predictions}
            aug_json_path = output_json_dir / f"{aug_id}.json"
            with open(aug_json_path, "w") as f:
                json.dump(aug_json, f, indent=2)

    print(f"Augmented data saved to: {output_img_dir} and {output_json_dir}")


def augment_dataset(image_dir: Path, output_dir: Path, config: dict) -> None:
    """
    Orchestrates augmentation pipeline.
    image_dir: path to labeled images
    output_dir: root output directory (will contain 'images' and 'labels')
    config: dictionary of augmentation settings
    """
    num_augmentations = config.get("num_augmentations", 3)

    labeled_json_dir = Path("mock_io/data/labeled")
    output_img_dir = output_dir / "images"
    output_json_dir = output_dir / "labels"

    # Match .json to corresponding image file (by stem)
    json_files = list(labeled_json_dir.glob("*.json"))
    image_files = [f for f in image_dir.glob("*") if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    image_lookup = {f.stem.lower(): f for f in image_files}
    matched_pairs = [
        (json_file, image_lookup[json_file.stem.lower()])
        for json_file in json_files
        if json_file.stem.lower() in image_lookup
    ]   

    # Apply augmentation
    transform = build_augmentation_transform(config)
    augment_images(matched_pairs, transform, output_img_dir, output_json_dir, num_augmentations)

    print(f"Found {len(json_files)} label files")
    print(f"Found {len(image_lookup)} image stems")
    print(f"Matched {len(matched_pairs)} json-image pairs")