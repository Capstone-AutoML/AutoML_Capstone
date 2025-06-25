# Augmentation

This module applies a range of image augmentations to the labeled dataset to improve model generalization during training. It uses the [Albumentations](https://albumentations.ai/docs/) library for robust image transformations while keeping bounding boxes aligned.

---

## Key Features

- Applies a configurable set of augmentations, including horizontal flip, brightness/contrast adjustment, color shift, noise, rotation, grayscale, and blur.
- Maintains bounding box alignment using `pascal_voc` format
- Saves augmented images and prediction labels in parallel
- Detects and handles images without predictions to avoid generating invalid annotations.

---

## Core Components

### `build_augmentation_transform(config: dict) -> A.Compose`

Creates an augmentation pipeline from a config dictionary.

Transforms include:

- `HorizontalFlip` (default `p=0.5`)
- `RandomBrightnessContrast` (default `p=0.5`)
- `HueSaturationValue` (default `p=0.5`)
- `Blur`(default `p=0.3` and `blur_limit=3`)
- `GaussNoise` (default `p=0.3`, min=10, max=50 )
- `ToGray` (default `p=0.2`)
- `Rotate` (default `p=0.4` and `rotate_limit=15`)

All parameters and probabilities are configurable.
> **Note:** YOLO uses upright bounding boxes for training. Modifying `rotate_limit` to larger angle may change the size of the bounding boxes and alter its accuracy.  


---

### `augment_images(matched_pairs: list, transform: A.Compose, output_img_dir: Path, output_json_dir: Path, num_augmentations: int,config: dict)`

Applies the transform pipeline on each image-label pair.

#### Inputs:
- `matched_pairs`: list of `(json_path, image_path)` tuples
- `transform`: Albumentations `Compose` object
- `output_img_dir`: Directory to save augmented `.jpg` images
- `output_json_dir`: Directory to save corresponding `.json` label files
- `num_augmentations`: Number of augmented versions to generate per image
- `config`: Dictionary that may include a base `"seed"` key for reproducibility.

#### Behavior:
- Saves augmented images as `.jpg` files and `.json` labels with matching filenames
- Handles images with no predictions by saving them unmodified to `no_prediction_images/`
- If a base seed is provided in `config`, offsets it by iteration index (`base_seed + i * 2`) to ensure consistent varied results across multiple augmentations per image. Essentially, this avoids applying the exact same augmentation when `num_augmentations` > 1.

---

### `augment_dataset(image_dir: Path, output_dir: Path, config: dict) -> None`

Coordinates the augmentation process end-to-end.

#### Inputs:
- `image_dir`: Directory containing the original input images  
- `output_dir`: Root directory where augmented `images/` and `labels/` will be saved  
- `config`: Dictionary of augmentation settings, including:
  - `num_augmentations`: Number of times each image should be augmented  
  - Transform parameters (e.g., probabilities and limits for each augmentation)  
  - `labeled_dir`: Path to the directory containing `.json` label files 

> **Note:** If the `image_dir` is modified, the `labeled_dir` in `augmentation_config.json` should also point to the matching directory that holds the corresponding `.json` label files.  
> By default, `labeled_dir` is set to `automl_workspace/data_pipeline/labeled`.

#### Workflow:
1. Loads all `.json` label files from the directory specified by `config["labeled_dir"]`  
2. Loads image files from `image_dir` and matches them with labels by filename stem  
3. Builds the augmentation transform using `build_augmentation_transform(config)`  
4. Calls `augment_images()` to apply the transform and save augmented outputs:
   - Augmented images are saved to `<output_dir>/images/` as `.jpg` files  
   - Corresponding augmented labels are saved to `<output_dir>/labels/` as `.json` files  
   - Original images without any predictions are saved (unmodified) to a separate folder: `<output_dir>/../no_prediction_images/`  
5. Prints a summary of:
   - Total label files loaded  
   - Total image files loaded  
   - Number of matched image-label pairs processed  
   - Output directories used for augmented files

---

## Configuration Parameters (for Augmentation from `pipeline_config.json`)

The following fields from the `augmentation_config.json` file directly control the **image augmentation pipeline**:

 | **Key**                      | **Description**                                                                 |
 |-----------------------------|---------------------------------------------------------------------------------|
 | `num_augmentations`         | Number of augmented versions to generate per image (default: `3`).             |
 | `horizontal_flip_prob`      | Probability of flipping the image horizontally (default: `0.5`).               |
 | `brightness_contrast_prob`  | Probability of applying brightness/contrast change (default: `0.5`).           |
 | `hue_saturation_prob`       | Probability of adjusting hue and saturation (default: `0.5`).                  |
 | `blur_prob`                 | Probability of applying Gaussian blur (default: `0.3`).                        |
 | `blur_limit`                | Maximum kernel size for blur (default: `3`).                                   |
 | `gauss_noise_prob`          | Probability of adding Gaussian noise (default: `0.3`).                         |
 | `gauss_noise_var_min`       | Minimum variance for Gaussian noise (default: `10.0`).                         |
 | `gauss_noise_var_max`       | Maximum variance for Gaussian noise (default: `50.0`).                         |
 | `grayscale_prob`            | Probability of converting the image to grayscale (default: `0.2`).             |
 | `rotate_prob`               | Probability of rotating the image (default: `0.4`).                            |
 | `rotate_limit`              | Maximum rotation angle in degrees (default: `15`).                             |

These values define how aggressively and in what ways the dataset will be augmented to improve model robustness.

---


## Example Call

```python
augment_dataset(
    image_dir=Path("automl_workspace/data_pipeline/input"),
    output_dir=Path("automl_workspace/data_pipeline/labeled"),
    config=config
)
```

---

## Output Structure

```
automl_workspace/
├── data_pipeline/
│   ├── labeled/                     # Original labels
│   ├── input/                       # Original images
│   ├── augmented/
│   │   ├── images/                  # Augmented image files
│   │   ├── labels/                  # Augmented JSON files
│   │   ├── no_prediction_images/    # Skipped originals with no predictions
```