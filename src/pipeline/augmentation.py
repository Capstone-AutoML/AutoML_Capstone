import cv2
import json
from pathlib import Path
from PIL import Image
import albumentations as A
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# Create augmentation pipeline
def build_augmentation_transform(config: dict) -> A.Compose:
    """
    Build the augmentation transform pipeline from the given configuration.

    This function constructs an Albumentations Compose object with a sequence of 
    image augmentation transforms. Each transform is applied with a configurable 
    probability and parameter set drawn from the `config` dictionary.

    Supported transforms:
    - HorizontalFlip
    - RandomBrightnessContrast
    - HueSaturationValue
    - Blur
    - GaussNoise
    - ToGray
    - Rotate

    The transform also ensures bounding box alignment using 'pascal_voc' format.

    Args:
        config (dict): Dictionary containing probability values and parameters 
                       for each augmentation transform.

    Returns:
        A.Compose: An Albumentations Compose object with the specified transformations 
                   and bounding box handling.
    """
    return A.Compose(
        [
        # Flips image horizontally, applied 50% of the time by default
        A.HorizontalFlip(p=config.get("horizontal_flip_prob", 0.5)),
        # Alters brightness and contrast, applied 50% of the time by default 
        A.RandomBrightnessContrast(p=config.get("brightness_contrast_prob", 0.5)),
        # Distorts colors in image
        A.HueSaturationValue(p=config.get("hue_saturation_prob", 0.5)),
        # Blurs image, kernel size (blur_limit=3) is 3x3
        A.Blur(blur_limit=config.get("blur_limit", 3), 
               p=config.get("blur_prob", 0.3)),
        # Adds noise to image, intensity ranges from 10-50
        A.GaussNoise(var_limit=(config.get("gauss_noise_var_min", 10.0), config.get("gauss_noise_var_max", 50.0)),
                      p=config.get("gauss_noise_prob", 0.3)),
        # Converts image to grayscale
        A.ToGray(p=config.get("grayscale_prob", 0.2)),
        # Rotates image up to 15 degrees, fills empty borders with black
        A.Rotate(limit=config.get("rotate_limit", 15), 
                 border_mode=cv2.BORDER_CONSTANT, 
                 p=config.get("rotate_prob", 0.4)),
    ], 
    bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])
    )


def augment_images(matched_pairs: list, 
                   transform: A.Compose, 
                   output_img_dir: Path, 
                   output_json_dir: Path, 
                   num_augmentations: int,
                   config: dict
    ) -> None:
    """
    Apply augmentations to each labeled image and save the results.

    For each (image, label) pair, this function applies the given transformation 
    pipeline `num_augmentations` times. It saves the augmented images and their 
    updated prediction labels (in JSON format) to the specified output directories.

    If an image has no predictions (empty bounding box list), the original image 
    is saved separately in a dedicated 'no_prediction_images' folder.

    Args:
        matched_pairs (list): List of tuples, each containing a Path to a JSON file 
                              and its corresponding image file.
        transform (A.Compose): Albumentations transformation pipeline.
        output_img_dir (Path): Directory to save augmented images.
        output_json_dir (Path): Directory to save augmented label files.
        num_augmentations (int): Number of times to apply augmentations per image.
        config (dict): Configuration dictionary that may include a base random seed.

    Returns:
        None
    """
    # Separate folder for un-augmented no-prediction images and labels
    no_pred_img_dir = output_img_dir.parent / "no_prediction_images"

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_json_dir.mkdir(parents=True, exist_ok=True)
    no_pred_img_dir.mkdir(parents=True, exist_ok=True)

    # Get base seed from augmentation config file
    base_seed = config.get("seed", None)

    for json_path, image_path in matched_pairs:
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with open(json_path, "r") as f:
            data = json.load(f)

        bboxes = [obj["bbox"] for obj in data["predictions"]]
        class_labels = [obj["class"] for obj in data["predictions"]]
        confidences = [obj["confidence"] for obj in data["predictions"]]

        if not bboxes:
            # Save original image to the no_prediction_images folder
            no_aug_image_path = no_pred_img_dir / image_path.name
            Image.fromarray(image).save(no_aug_image_path)

            continue

        for i in range(num_augmentations):

            # If base seed is set, adjust for iteration
            if base_seed is not None:
                transform.set_random_seed(base_seed + i * 2)

            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_classes = augmented["class_labels"]

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

            aug_json = {"predictions": aug_predictions}
            aug_json_path = output_json_dir / f"{aug_id}.json"
            with open(aug_json_path, "w") as f:
                json.dump(aug_json, f, indent=2)

    print(f"Augmented images saved to: {output_img_dir}")
    print(f"No-prediction images saved to: {no_pred_img_dir}")
    print(f"Augmented labels saved to: {output_json_dir}")


def augment_dataset(image_dir: Path, output_dir: Path, config: dict) -> None:
    """
    Orchestrates the full augmentation pipeline for a labeled image dataset.

    This function matches labeled JSON files with their corresponding images,
    builds the augmentation pipeline from the provided config, and applies
    augmentations using `augment_images`.

    Args:
        image_dir (Path): Directory containing the original labeled images.
        output_dir (Path): Root directory where augmented 'images/' and 'labels/' will be saved.
        config (dict): Dictionary containing augmentation settings, including
                       number of augmentations and optional transform parameters.

    Behavior:
        - Loads label files from a `labeled_json_dir` (‘automl_workspace/data_pipeline/labeled’)
        - Matches JSON labels to image files by filename stem
        - Builds an Albumentations transform pipeline using `build_augmentation_transform`
        - Applies the transform using `augment_images` with `num_augmentations` per image
        - Logs counts of label files, image files, and successful matches

    Returns:
        None
    """
    num_augmentations = config.get("num_augmentations", 3)

    labeled_json_dir = Path(config.get("label_dir", "automl_workspace/data_pipeline/labeled"))
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

    transform = build_augmentation_transform(config)
    augment_images(matched_pairs, transform, output_img_dir, output_json_dir, num_augmentations, config)

    print(f"Found {len(json_files)} label files")
    print(f"Found {len(image_lookup)} image stems")
    print(f"Matched {len(matched_pairs)} json-image pairs")