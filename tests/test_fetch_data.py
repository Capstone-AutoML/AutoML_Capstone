"""
Tests for the fetch_data module.
"""

import pytest
from pathlib import Path
import shutil
import os

from src.pipeline.fetch_data import (
    _create_directory_structure,
    _load_images,
    _save_images_to_raw,
    _sample_and_save_distilled,
    fetch_and_organize_images
)

@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    source_dir = tmp_path / "source"
    raw_dir = tmp_path / "raw"
    distilled_dir = tmp_path / "distilled"
    
    # Create source directory with some test images
    source_dir.mkdir()
    for i in range(5):
        (source_dir / f"test_image_{i}.jpg").touch()
        (source_dir / f"test_image_{i}.png").touch()
    
    yield source_dir, raw_dir, distilled_dir
    
    # Cleanup
    shutil.rmtree(tmp_path)

def test_create_directory_structure(temp_dirs):
    """Test directory structure creation."""
    _, raw_dir, distilled_dir = temp_dirs
    
    _create_directory_structure(raw_dir, distilled_dir)
    
    assert raw_dir.exists()
    assert distilled_dir.exists()
    assert raw_dir.is_dir()
    assert distilled_dir.is_dir()

def test_load_images(temp_dirs):
    """Test loading images from source directory."""
    source_dir, _, _ = temp_dirs
    
    images = _load_images(source_dir)
    
    assert len(images) == 10  # 5 jpg + 5 png files
    assert all(img.suffix.lower() in {'.jpg', '.png'} for img in images)
    assert all(img.exists() for img in images)

def test_save_images_to_raw(temp_dirs):
    """Test saving images to raw directory."""
    source_dir, raw_dir, _ = temp_dirs
    
    # Create raw directory
    raw_dir.mkdir()
    
    # Load and save images
    images = _load_images(source_dir)
    _save_images_to_raw(images, raw_dir)
    
    # Check if all images were copied
    raw_images = list(raw_dir.glob('*'))
    assert len(raw_images) == len(images)
    assert all(img.exists() for img in raw_images)

def test_sample_and_save_distilled(temp_dirs):
    """Test sampling and saving distilled images."""
    source_dir, _, distilled_dir = temp_dirs
    
    # Create distilled directory
    distilled_dir.mkdir()
    
    # Load images
    images = _load_images(source_dir)
    
    # Test with integer sample size
    num_sampled = _sample_and_save_distilled(images, distilled_dir, 3, seed=42)
    assert num_sampled == 3
    assert len(list(distilled_dir.glob('*'))) == 3
    
    # Clean distilled directory
    for file in distilled_dir.glob('*'):
        file.unlink()
    
    # Test with float sample size
    num_sampled = _sample_and_save_distilled(images, distilled_dir, 0.5, seed=42)
    assert num_sampled == 5  # 50% of 10 images
    assert len(list(distilled_dir.glob('*'))) == 5

def test_fetch_and_organize_images(temp_dirs):
    """Test the main fetch_and_organize_images function."""
    source_dir, raw_dir, distilled_dir = temp_dirs
    
    # Test without distillation
    config = {"distillation_image_prop": 0}
    fetch_and_organize_images(source_dir, raw_dir, distilled_dir, config)
    
    assert raw_dir.exists()
    assert len(list(raw_dir.glob('*'))) == 10
    # Check that distilled directory exists but is empty
    assert distilled_dir.exists()
    assert len(list(distilled_dir.glob('*'))) == 0
    
    # Clean up
    shutil.rmtree(raw_dir)
    shutil.rmtree(distilled_dir)
    
    # Test with distillation
    config = {"distillation_image_prop": 0.3}
    fetch_and_organize_images(source_dir, raw_dir, distilled_dir, config, seed=42)
    
    assert raw_dir.exists()
    assert distilled_dir.exists()
    assert len(list(raw_dir.glob('*'))) == 10
    assert len(list(distilled_dir.glob('*'))) == 3  # 30% of 10 images

def test_invalid_source_directory():
    """Test behavior with invalid source directory."""
    # The function should return an empty list for non-existent directory
    images = _load_images(Path("nonexistent_directory"))
    assert len(images) == 0

def test_empty_source_directory(tmp_path):
    """Test behavior with empty source directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    images = _load_images(empty_dir)
    assert len(images) == 0 