"""
Script for fetching and organizing raw image data from local storage or cloud sources.
"""

import shutil
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def _sample_and_save_distilled(images: List[Path], distilled_dir: Path, sample_size: Union[int, float], seed: Optional[int] = 42) -> int:
    """
    Samples a subset of images and saves them to the distilled directory.
    
    Args:
        images (List[Path]): List of image paths to sample from
        distilled_dir (Path): Destination directory for distilled images
        sample_size (Union[int, float]): Number of images to sample (int) or proportion (float)
        seed (Optional[int]): Random seed for reproducibility
        
    Returns:
        int: Number of images actually sampled and saved
    """
    random.seed(seed)
    
    # Calculate actual sample size based on type
    if isinstance(sample_size, float):
        actual_sample_size = int(len(images) * sample_size)
    else:
        actual_sample_size = min(sample_size, len(images))
        
    sampled_images = random.sample(images, actual_sample_size)
    for img_path in sampled_images:
        dest_path = distilled_dir / img_path.name
        shutil.copy2(img_path, dest_path)
        
    return actual_sample_size

def fetch_and_organize_images(source_dir: Path, raw_dir: Path, distilled_dir: Path, config: Dict, seed: Optional[int] = None) -> None:
    """
    Main function to fetch images from source, save to raw directory, and create a distilled subset.
    
    Args:
        source_dir (Path): Path to source directory containing images
        raw_dir (Path): Path to save all raw images
        distilled_dir (Path): Path to save sampled distilled images
        config (Dict): Configuration dictionary containing pipeline parameters
        seed (Optional[int]): Random seed for reproducibility
    """
    # Create necessary directories
    _create_directory_structure(raw_dir, distilled_dir)
    
    # Load all images
    images = _load_images(source_dir)
    print(f"Found {len(images)} images in source directory")
    
    # Save to raw directory
    _save_images_to_raw(images, raw_dir)
    print(f"Saved {len(images)} images to raw directory")
    
    # Sample and save to distilled directory if distillation is enabled
    if config.get("distillation_image_prop", 0) > 0:
        num_sampled = _sample_and_save_distilled(images, distilled_dir, config["distillation_image_prop"], seed)
        print(f"Sampled and saved {num_sampled} images to distilled directory")
    else:
        print("Skipping distillation set creation as per configuration") 