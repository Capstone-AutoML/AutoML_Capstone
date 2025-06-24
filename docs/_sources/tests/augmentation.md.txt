# Augmentation

This page outlines the unit test coverage for the `augmentation.py` module, which applies configurable image augmentations to labeled datasets using Albumentations. It ensures the generation of augmented images and updated label JSONs while preserving bounding box structure.

---

## Coverage Overview

This suite tests the following components:

- `build_augmentation_transform`: Builds the augmentation pipeline from a configuration dictionary.
- `augment_images`: Applies augmentations to matched image-label pairs.
- `augment_dataset`: Full pipeline integration test using a real image and JSON file.

---

## Fixtures

### `sample_config`  
Provides a valid augmentation configuration dictionary with all probabilities set to 1.0 to force all transforms to apply. Includes:

- Horizontal flip
- Brightness/contrast
- Hue/saturation
- Gaussian blur and noise
- Grayscale
- Rotation

### `temp_image_and_json`  
Creates a temporary test image and its corresponding JSON label in a valid format:
- Bounding box: `[10, 10, 50, 50]`
- Class: `"FireBSI"`
- Confidence: `0.9`

---

## Function Tests

### `build_augmentation_transform`

- `test_build_augmentation_transform`:  
  Validates that the transform pipeline builds successfully and is callable.

---

### `augment_images`

- `test_augment_images_creates_augmented_files`:  
  Verifies that the correct number of augmented images and labels are created based on `num_augmentations`.  
  Also checks that each output JSON contains a valid `predictions` list.

---

### `augment_dataset`

- `test_augment_dataset_integration`:  
  Integration test that simulates a full run of the `augment_dataset()` pipeline.
  
  It:
  - Copies the image to an input directory
  - Creates a valid labeled directory containing its JSON label
  - Runs the pipeline
  - Verifies that augmented images and labels are created in the output directory

---

## Summary

This page outlines a focused test suite ensuring:

- Augmentation configurations are valid and interpretable
- Image-label pairs are augmented correctly and written to disk
- End-to-end augmentation logic functions as expected in a simulated pipeline environment

These tests confirm that the augmentation module integrates safely and predictably with the rest of the AutoML pipeline.
