"""
Script for fetching and organizing raw image data from local storage or cloud sources.
"""

import shutil
import random
from pathlib import Path
from typing import List, Tuple

def _create_directory_structure(raw_dir: Path, distilled_dir: Path) -> None:
    """
    Creates necessary directories for raw and distilled images if they don't exist.
    
    Args:
        raw_dir (Path): Path to raw images directory
        distilled_dir (Path): Path to distilled images directory
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    distilled_dir.mkdir(parents=True, exist_ok=True)

def _load_images(source_dir: Path) -> List[Path]:
    """
    Loads all image files from the source directory.
    
    Args:
        source_dir (Path): Path to directory containing images
        
    Returns:
        List[Path]: List of paths to image files
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    return [f for f in source_dir.glob('**/*') if f.suffix.lower() in image_extensions]

def _save_images_to_raw(images: List[Path], raw_dir: Path) -> None:
    """
    Copies images to the raw images directory.
    
    Args:
        images (List[Path]): List of image paths to copy
        raw_dir (Path): Destination directory for raw images
    """
    for img_path in images:
        dest_path = raw_dir / img_path.name
        shutil.copy2(img_path, dest_path)

def _sample_and_save_distilled(images: List[Path], distilled_dir: Path, sample_size: int = 100, seed: int = 42) -> None:
    """
    Samples a subset of images and saves them to the distilled directory.
    
    Args:
        images (List[Path]): List of image paths to sample from
        distilled_dir (Path): Destination directory for distilled images
        sample_size (int): Number of images to sample
        seed (int, optional): Random seed for reproducibility
    """
    random.seed(seed)
    sampled_images = random.sample(images, min(sample_size, len(images)))
    for img_path in sampled_images:
        dest_path = distilled_dir / img_path.name
        shutil.copy2(img_path, dest_path)

def fetch_and_organize_images(source_dir: Path, raw_dir: Path, distilled_dir: Path, sample_size: int = 100, seed: int = None) -> None:
    """
    Main function to fetch images from source, save to raw directory, and create a distilled subset.
    
    Args:
        source_dir (Path): Path to source directory containing images
        raw_dir (Path): Path to save all raw images
        distilled_dir (Path): Path to save sampled distilled images
        sample_size (int): Number of images to sample for distilled set
        seed (int, optional): Random seed for reproducibility
    """
    # Create necessary directories
    _create_directory_structure(raw_dir, distilled_dir)
    
    # Load all images
    images = _load_images(source_dir)
    print(f"Found {len(images)} images in source directory")
    
    # Save to raw directory
    _save_images_to_raw(images, raw_dir)
    print(f"Saved {len(images)} images to raw directory")
    
    # Sample and save to distilled directory
    _sample_and_save_distilled(images, distilled_dir, sample_size, seed)
    print(f"Sampled and saved {sample_size} images to distilled directory") 