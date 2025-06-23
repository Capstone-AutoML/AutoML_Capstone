"""
Tests for utility functions.
"""
import os
import pytest
import json
from unittest.mock import patch
from unittest.mock import mock_open
from src.utils import (
    detect_device,
    load_config,
    create_data_yaml,
    create_quantize_yaml
)


@patch('torch.cuda.is_available', return_value=True)
def test_detect_device_cuda(mock_cuda):
    assert detect_device() == "cuda"


@patch('torch.cuda.is_available', return_value=False)  
@patch('torch.backends.mps.is_available', return_value=True)
def test_detect_device_mps(mock_mps, mock_cuda):
    assert detect_device() == "mps"


@patch('torch.cuda.is_available', return_value=False)
@patch('torch.backends.mps.is_available', return_value=False)
def test_detect_device_cpu(mock_mps, mock_cuda):
    assert detect_device() == "cpu"


# Config loading tests
def test_load_config_valid_file(tmp_path):
    """Test loading a valid config file."""
    config_file = tmp_path / "config.json"
    test_config = {"key": "value", "distillation_image_prop": 0.5}

    with open(config_file, 'w') as f:
        json.dump(test_config, f)

    result = load_config(config_file)
    assert result == test_config


def test_load_config_file_not_found(tmp_path):
    """Test handling of missing config file."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent.json")


def test_load_config_invalid_json(tmp_path):
    """Test handling of invalid JSON."""
    config_file = tmp_path / "config.json"
    config_file.write_text("invalid json content")

    with pytest.raises(json.JSONDecodeError):
        load_config(config_file)


def test_load_config_invalid_distillation_prop(tmp_path):
    """Test validation of distillation_image_prop."""
    config_file = tmp_path / "config.json"
    test_config = {"distillation_image_prop": -0.5}

    with open(config_file, 'w') as f:
        json.dump(test_config, f)

    with pytest.raises(ValueError, match="cannot be negative"):
        load_config(config_file)


def test_load_config_default_path():
    """Test loading config from default path."""
    with patch('pathlib.Path.exists', return_value=True):
        with patch('builtins.open', mock_open(read_data='{"key": "value"}')):
            result = load_config()
            assert result == {"key": "value"}


def test_create_data_yaml(tmp_path):
    """Test creating data.yaml file."""
    create_data_yaml(str(tmp_path))

    # Check if the data.yaml file was created
    yaml_file = tmp_path / "data.yaml"
    assert yaml_file.exists()

    # Verify basic content
    content = yaml_file.read_text()
    assert "train:" in content
    assert "val:" in content
    assert "nc:" in content


def test_create_quantize_yaml(tmp_path):
    """Test creating quantize.yaml file."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        create_quantize_yaml(str(tmp_path))

        # Check if the quantize.yaml file was created
        yaml_file = tmp_path / "src" / "quantize.yaml"
        assert yaml_file.exists()

        # Verify basic content
        content = yaml_file.read_text()
        assert "nc:" in content
        assert "names:" in content
    finally:
        os.chdir(original_cwd)
