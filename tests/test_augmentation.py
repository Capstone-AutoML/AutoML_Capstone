"""
Unit tests for augmentation module.
"""

import pytest
import shutil
import json
import numpy as np
from PIL import Image
from pathlib import Path

import cv2
from src.pipeline.augmentation import (
    build_augmentation_transform,
    augment_images,
    augment_dataset
)


@pytest.fixture
def sample_config():
    """Returns a sample augmentation config."""
    return {
        "num_augmentations": 2,
        "horizontal_flip_prob": 1.0,
        "brightness_contrast_prob": 1.0,
        "hue_saturation_prob": 1.0,
        "blur_prob": 1.0,
        "blur_limit": 3,
        "gauss_noise_prob": 1.0,
        "gauss_noise_var_min": 10,
        "gauss_noise_var_max": 20,
        "grayscale_prob": 1.0,
        "rotate_prob": 1.0,
        "rotate_limit": 10,
    }


@pytest.fixture
def temp_image_and_json(tmp_path):
    """Create a single image and JSON pair in the temp directory."""
    image_dir = tmp_path / "images"
    label_dir = tmp_path / "labels"
    image_dir.mkdir()
    label_dir.mkdir()

    image_path = image_dir / "test_image.jpg"
    label_path = label_dir / "test_image.json"

    # Create dummy image
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), dummy_img)

    # Create dummy JSON label
    dummy_label = {
        "predictions": [
            {"bbox": [10, 10, 50, 50], "confidence": 0.9, "class": "FireBSI"}
        ]
    }

    with open(label_path, "w") as f:
        json.dump(dummy_label, f)

    return [(label_path, image_path)]


def test_build_augmentation_transform(sample_config):
    """Test if the transform builds successfully."""
    transform = build_augmentation_transform(sample_config)
    assert transform is not None
    assert hasattr(transform, "__call__")


def test_augment_images_creates_augmented_files(tmp_path, sample_config, temp_image_and_json):
    """Test augment_images creates expected image and label files."""
    output_img_dir = tmp_path / "augmented_images"
    output_json_dir = tmp_path / "augmented_labels"
    matched_pairs = temp_image_and_json
    transform = build_augmentation_transform(sample_config)

    augment_images(matched_pairs, transform, output_img_dir, output_json_dir, num_augmentations=2)

    images = list(output_img_dir.glob("*.jpg"))
    labels = list(output_json_dir.glob("*.json"))

    assert len(images) == 2  # There are 2 augmentations
    assert len(labels) == 2

    for img_path in images:
        assert img_path.exists()

    for label_path in labels:
        with open(label_path, "r") as f:
            data = json.load(f)
            assert "predictions" in data
            assert isinstance(data["predictions"], list)


def test_augment_dataset_integration(tmp_path, sample_config, temp_image_and_json):
    """Test the full augmentation pipeline using augment_dataset."""
    image_dir = tmp_path / "input_images"
    output_dir = tmp_path / "aug_output"

    # Move the test image to mock input location
    image_dir.mkdir()
    shutil.copy(temp_image_and_json[0][1], image_dir / "test_image.jpg")

    # Patch mock_io/data/labeled to point to the label file
    mock_labeled = Path("mock_io/data/labeled")
    mock_labeled.mkdir(parents=True, exist_ok=True)
    shutil.copy(temp_image_and_json[0][0], mock_labeled / "test_image.json")

    augment_dataset(image_dir, output_dir, sample_config)

    assert (output_dir / "images").exists()
    assert (output_dir / "labels").exists()

    assert len(list((output_dir / "images").glob("*.jpg"))) >= 1
    assert len(list((output_dir / "labels").glob("*.json"))) >= 1
