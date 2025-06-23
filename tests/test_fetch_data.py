"""
Tests for the fetch_data module.
"""

import pytest
from pathlib import Path
import shutil

from src.pipeline.fetch_data import validate_input_images


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    source_dir = tmp_path / "source"

    # Create source directory with some test images
    source_dir.mkdir()
    for i in range(5):
        (source_dir / f"test_image_{i}.jpg").touch()
        (source_dir / f"test_image_{i}.png").touch()

    yield source_dir

    # Cleanup
    shutil.rmtree(tmp_path, ignore_errors=True)


def test_validate_input_images_success(temp_dirs):
    """Test validating images when images exist."""
    source_dir = temp_dirs

    # Should not raise an exception
    validate_input_images(source_dir)


def test_validate_input_images_empty_directory(tmp_path):
    """Test validating images with empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Should raise ValueError
    with pytest.raises(ValueError, match="No images found in input directory"):
        validate_input_images(empty_dir)


def test_validate_input_images_no_valid_images(tmp_path):
    """Test validating directory with no valid image files."""
    dir_with_text = tmp_path / "text_only"
    dir_with_text.mkdir()

    (dir_with_text / "readme.txt").touch()
    (dir_with_text / "data.csv").touch()
    (dir_with_text / "config.json").touch()

    # Non-image files should raise ValueError
    with pytest.raises(ValueError, match="No images found in input directory"):
        validate_input_images(dir_with_text)


def test_validate_input_images_mixed_files(tmp_path):
    """Test validating directory with mix of image and non-image files."""
    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()

    (mixed_dir / "image1.jpg").touch()
    (mixed_dir / "image2.png").touch()
    (mixed_dir / "readme.txt").touch()
    (mixed_dir / "data.csv").touch()

    # Images exist, should not raise an exception
    validate_input_images(mixed_dir)


def test_validate_input_images_different_extensions(tmp_path):
    """Test validating images with different supported extensions."""
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    (image_dir / "image.jpg").touch()
    (image_dir / "image.jpeg").touch()
    (image_dir / "image.png").touch()
    (image_dir / "image.bmp").touch()
    (image_dir / "image.tiff").touch()
    (image_dir / "image.gif").touch()

    # Images supported, should not raise an exception
    validate_input_images(image_dir)


def test_validate_input_images_nonexistent_directory():
    """Test validating nonexistent directory."""
    nonexistent_dir = Path("nonexistent_directory")

    # Directory not exist, should raise ValueError
    with pytest.raises(ValueError, match="No images found in input directory"):
        validate_input_images(nonexistent_dir)
