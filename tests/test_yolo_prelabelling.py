"""
Full test suite for `yolo_prelabelling.py`.

Covers:
- _load_model
- _get_image_files
- _process_prediction
- _save_predictions
- generate_yolo_prelabelling (normal and edge cases)
- device='auto' fallback
- corrupted/unreadable images
- model loading errors
"""

import pytest
import numpy as np
import cv2
import json
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from src.pipeline.prelabelling.yolo_prelabelling import (
    generate_yolo_prelabelling,
    _get_image_files,
    _load_model,
    _process_prediction,
    _save_predictions
)


class DummyYOLOModel:
    """Mock YOLO model for testing."""
    
    def __init__(self):
        self.names = {0: 'fire', 1: 'smoke'}
    
    def to(self, device):
        return self
    
    def __call__(self, image_path, verbose=False):
        # Return a mock result with dummy predictions
        result = Mock()
        result.boxes = Mock()
        
        # Create mock boxes with dummy data
        box1 = Mock()
        box1.xyxy = [np.array([10, 10, 40, 40])]
        box1.conf = [np.array([0.8])]
        box1.cls = [np.array([0])]
        
        box2 = Mock()
        box2.xyxy = [np.array([50, 50, 80, 80])]
        box2.conf = [np.array([0.6])]
        box2.cls = [np.array([1])]
        
        result.boxes = [box1, box2]
        result.names = self.names
        
        return [result]


@pytest.fixture
def patch_yolo_model(monkeypatch):
    """Patch YOLO model to return a dummy model."""
    def mock_yolo(*args, **kwargs):
        return DummyYOLOModel()
    
    monkeypatch.setattr("src.pipeline.prelabelling.yolo_prelabelling.YOLO", mock_yolo)


@pytest.fixture
def tmp_dirs_with_images(tmp_path):
    """Create temporary directories with test images."""
    raw = tmp_path / "raw"
    out = tmp_path / "out"
    raw.mkdir()
    out.mkdir()
    
    # Create valid test images
    dummy_img1 = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(raw / "img1.jpg"), dummy_img1)
    
    dummy_img2 = 128 * np.ones((150, 150, 3), dtype=np.uint8)
    cv2.imwrite(str(raw / "img2.png"), dummy_img2)
    
    # Create a corrupted image
    (raw / "corrupted.jpg").write_bytes(b"\x00\x11\x22\x33")
    
    # Create a non-image file
    (raw / "text.txt").write_text("This is not an image")
    
    return raw, out


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a mock model path."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"dummy model data")
    return model_path


def test_get_image_files_valid(tmp_dirs_with_images):
    """
    Test that _get_image_files correctly finds all image files in a directory.
    This test ensures that only files with valid image extensions are returned, including corrupted images (since extension is the only filter).
    It verifies that the function does not include non-image files in its output.
    """
    raw, _ = tmp_dirs_with_images
    files = _get_image_files(raw)
    
    assert len(files) == 3  # img1.jpg, img2.png, corrupted.jpg
    assert all(f.suffix.lower() in {".jpg", ".jpeg", ".png"} for f in files)
    assert any(f.name == "img1.jpg" for f in files)
    assert any(f.name == "img2.png" for f in files)


def test_get_image_files_empty_directory(tmp_path):
    """
    Test that _get_image_files returns an empty list for an empty directory.
    This ensures the function does not fail or return unexpected results when no files are present.
    """
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    files = _get_image_files(empty_dir)
    assert len(files) == 0


def test_load_model_success(mock_model_path):
    """
    Test that _load_model successfully loads a YOLO model from a valid path.
    This test checks that the model is loaded and moved to the correct device, and that the correct arguments are passed to the YOLO constructor.
    """
    with patch("src.pipeline.prelabelling.yolo_prelabelling.YOLO") as mock_yolo:
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        result = _load_model(mock_model_path, "cpu")
        
        mock_yolo.assert_called_once_with(str(mock_model_path))
        mock_model.to.assert_called_once_with("cpu")
        assert result == mock_model


def test_load_model_file_not_found(tmp_path):
    """
    Test that _load_model raises FileNotFoundError for a missing model file.
    This ensures the function fails fast and clearly when the model path does not exist, preventing silent errors downstream.
    """
    non_existent_path = tmp_path / "nonexistent.pt"
    
    with pytest.raises(FileNotFoundError, match="Model not found"):
        _load_model(non_existent_path, "cpu")


def test_process_prediction():
    """
    Test that _process_prediction correctly parses YOLO prediction results.
    This test checks that bounding boxes, confidence scores, and class names are extracted and formatted as expected for multiple detections.
    """
    # Create mock result
    result = Mock()
    result.names = {0: 'fire', 1: 'smoke'}
    
    # Create mock boxes
    box1 = Mock()
    box1.xyxy = [np.array([10, 10, 40, 40])]
    box1.conf = [np.array([0.8])]
    box1.cls = [np.array([0])]
    
    box2 = Mock()
    box2.xyxy = [np.array([50, 50, 80, 80])]
    box2.conf = [np.array([0.6])]
    box2.cls = [np.array([1])]
    
    result.boxes = [box1, box2]
    
    predictions = _process_prediction(result)
    
    assert len(predictions) == 2
    assert predictions[0]['bbox'] == [10, 10, 40, 40]
    assert predictions[0]['confidence'] == 0.8
    assert predictions[0]['class'] == 'fire'
    assert predictions[1]['bbox'] == [50, 50, 80, 80]
    assert predictions[1]['confidence'] == 0.6
    assert predictions[1]['class'] == 'smoke'


def test_process_prediction_empty():
    """
    Test that _process_prediction returns an empty list when there are no detections.
    This ensures the function handles the edge case of no predictions gracefully, returning an empty result instead of failing.
    """
    result = Mock()
    result.boxes = []
    result.names = {}
    
    predictions = _process_prediction(result)
    assert len(predictions) == 0


def test_save_predictions(tmp_path):
    """
    Test that _save_predictions writes predictions to a JSON file in the correct format.
    This test ensures the output file exists, is valid JSON, and contains the expected prediction data structure.
    """
    predictions = [
        {'bbox': [10, 10, 40, 40], 'confidence': 0.8, 'class': 'fire'},
        {'bbox': [50, 50, 80, 80], 'confidence': 0.6, 'class': 'smoke'}
    ]
    
    output_path = tmp_path / "predictions.json"
    _save_predictions(predictions, output_path)
    
    assert output_path.exists()
    
    with open(output_path, 'r') as f:
        saved_data = json.load(f)
    
    assert 'predictions' in saved_data
    assert len(saved_data['predictions']) == 2
    assert saved_data['predictions'] == predictions


def test_generate_yolo_prelabelling_success(patch_yolo_model, tmp_dirs_with_images, mock_model_path):
    """
    Test that generate_yolo_prelabelling processes all image files and creates JSON outputs.
    This test checks that the function iterates over all images, including corrupted ones (since the mock model always succeeds), and writes the expected number of output files.
    It also verifies the structure of the output JSON.
    """
    raw, out = tmp_dirs_with_images
    config = {"torch_device": "cpu"}
    
    generate_yolo_prelabelling(raw, out, mock_model_path, config)
    
    # Check that JSON files were created for valid images
    json_files = list(out.glob("*.json"))
    assert len(json_files) == 3  # Should process img1.jpg, img2.png, and corrupted.jpg (mock handles it)
    
    # Check content of one of the files
    with open(json_files[0], 'r') as f:
        data = json.load(f)
        assert 'predictions' in data
        assert len(data['predictions']) == 2  # Two detections per image


def test_generate_yolo_prelabelling_device_auto(patch_yolo_model, tmp_dirs_with_images, mock_model_path):
    """
    Test that generate_yolo_prelabelling uses detect_device when device is set to 'auto'.
    This ensures the device auto-detection logic is triggered and the model is loaded on the detected device.
    """
    raw, out = tmp_dirs_with_images
    
    with patch("src.pipeline.prelabelling.yolo_prelabelling.detect_device", return_value="cpu") as mock_detect:
        config = {"torch_device": "auto"}
        generate_yolo_prelabelling(raw, out, mock_model_path, config)
        
        mock_detect.assert_called_once()


def test_generate_yolo_prelabelling_handles_corrupted_images(patch_yolo_model, tmp_dirs_with_images, mock_model_path):
    """
    Test that generate_yolo_prelabelling does not fail on corrupted images when using the mock model.
    This test ensures that the pipeline is robust to unreadable or corrupted files, as the mock model always returns predictions.
    """
    raw, out = tmp_dirs_with_images
    config = {"torch_device": "cpu"}
    
    generate_yolo_prelabelling(raw, out, mock_model_path, config)
    
    # With mock model, all images (including corrupted) are processed successfully
    json_files = list(out.glob("*.json"))
    assert len(json_files) == 3


def test_generate_yolo_prelabelling_model_error(tmp_dirs_with_images, tmp_path):
    """
    Test that generate_yolo_prelabelling raises FileNotFoundError if the model file does not exist.
    This ensures the pipeline fails fast and clearly when the model path is invalid, preventing silent errors.
    """
    raw, out = tmp_dirs_with_images
    non_existent_model = tmp_path / "nonexistent.pt"
    config = {"torch_device": "cpu"}
    
    with pytest.raises(FileNotFoundError):
        generate_yolo_prelabelling(raw, out, non_existent_model, config)


def test_generate_yolo_prelabelling_empty_directory(patch_yolo_model, tmp_path, mock_model_path):
    """
    Test that generate_yolo_prelabelling does nothing when the input directory is empty.
    This ensures the function does not fail or create any output files when there are no images to process.
    """
    raw = tmp_path / "empty_raw"
    out = tmp_path / "empty_out"
    raw.mkdir()
    out.mkdir()
    
    config = {"torch_device": "cpu"}
    generate_yolo_prelabelling(raw, out, mock_model_path, config)
    
    # Should not create any JSON files
    json_files = list(out.glob("*.json"))
    assert len(json_files) == 0


def test_generate_yolo_prelabelling_verbose_mode(patch_yolo_model, tmp_dirs_with_images, mock_model_path, capsys):
    """
    Test that generate_yolo_prelabelling prints verbose output when verbose=True.
    This test checks that the expected log messages are printed, including device info, model loading, image count, and summary statistics.
    """
    raw, out = tmp_dirs_with_images
    config = {"torch_device": "cpu"}
    
    generate_yolo_prelabelling(raw, out, mock_model_path, config, verbose=True)
    
    captured = capsys.readouterr()
    assert "Using device: cpu" in captured.out
    assert "Loaded YOLO model" in captured.out
    assert "Found 3 images to process" in captured.out
    assert "Successfully processed: 3 images" in captured.out
    assert "Failed to process: 0 images" in captured.out


def test_generate_yolo_prelabelling_creates_output_directory(patch_yolo_model, tmp_path, mock_model_path):
    """
    Test that generate_yolo_prelabelling creates the output directory if it does not exist.
    This ensures the function is robust to missing output directories and does not fail due to missing paths.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    
    # Create a test image
    dummy_img = 255 * np.ones((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(raw / "test.jpg"), dummy_img)
    
    out = tmp_path / "new_output_dir"
    config = {"torch_device": "cpu"}
    
    generate_yolo_prelabelling(raw, out, mock_model_path, config)
    
    assert out.exists()
    assert out.is_dir()
    assert any(out.glob("*.json"))


def test_generate_yolo_prelabelling_pytorch_mps_fallback(patch_yolo_model, tmp_dirs_with_images, mock_model_path):
    """
    Test that generate_yolo_prelabelling sets the PYTORCH_ENABLE_MPS_FALLBACK environment variable.
    This ensures the pipeline is compatible with Apple Silicon and MPS fallback is enabled for PyTorch if needed.
    """
    raw, out = tmp_dirs_with_images
    config = {"torch_device": "cpu"}
    
    original_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    
    try:
        generate_yolo_prelabelling(raw, out, mock_model_path, config)
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"
    finally:
        if original_env is None:
            os.environ.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
        else:
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = original_env 