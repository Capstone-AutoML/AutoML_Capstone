# Augmentation Module

This module generates additional training data by applying randomized **image augmentations** to the labeled dataset. It uses the [Albumentations](https://albumentations.ai/docs/) library for robust image transformations while keeping bounding boxes aligned.

---

## Key Features

- Supports flips, brightness changes, noise, rotation, grayscale, and blur
- Maintains bounding box alignment using `pascal_voc` format
- Saves augmented images and prediction labels in parallel
- Handles images with no predictions separately

---

## Core Components

### `build_augmentation_transform(config: dict) -> A.Compose`

Creates an augmentation pipeline from a config dictionary.

Transforms include:

- `HorizontalFlip` (default `p=0.5`)
- `RandomBrightnessContrast`
- `HueSaturationValue`
- `Blur`
- `GaussNoise`
- `ToGray`
- `Rotate`

All parameters and probabilities are configurable.

---

### `augment_images(...)`

Applies the transform pipeline on each image-label pair.

#### Inputs:
- `matched_pairs`: list of `(json_path, image_path)` tuples
- `transform`: Albumentations Compose object
- `output_img_dir`: where augmented images will be saved
- `output_json_dir`: where labels will be saved
- `num_augmentations`: how many times to augment each image

#### Behavior:
- Saves augmented `.jpg` files and `.json` labels side by side
- Stores no-prediction images in `no_prediction_images/`

---

### `augment_dataset(...)`

Coordinates the augmentation process end-to-end.

#### Inputs:
- `image_dir`: input directory with raw images
- `output_dir`: directory where augmented `images/` and `labels/` go
- `config`: contains hyperparameters like `num_augmentations`

#### Workflow:
1. Loads all `.json` from `mock_io/data/labeled`
2. Matches them with image files by filename stem
3. Builds augmentation transform
4. Calls `augment_images()` to perform the pipeline
5. Prints processing summary

---

## Example Config

```python
config = {
    "num_augmentations": 3,
    "horizontal_flip_prob": 0.5,
    "brightness_contrast_prob": 0.5,
    "hue_saturation_prob": 0.5,
    "blur_limit": 3,
    "blur_prob": 0.3,
    "gauss_noise_var_min": 10.0,
    "gauss_noise_var_max": 50.0,
    "gauss_noise_prob": 0.3,
    "grayscale_prob": 0.2,
    "rotate_limit": 15,
    "rotate_prob": 0.4
}
```

---

## Example Call

```python
augment_dataset(
    image_dir=Path("mock_io/data/raw/images"),
    output_dir=Path("mock_io/data/augmented"),
    config=config
)
```

---

## Output Structure

```
mock_io/
├── data/
│   ├── labeled/                # Original labels
│   ├── raw/images/            # Original images
│   ├── augmented/
│   │   ├── images/            # Augmented image files
│   │   ├── labels/            # Augmented JSON files
│   └── no_prediction_images/  # Skipped originals with no predictions
```
