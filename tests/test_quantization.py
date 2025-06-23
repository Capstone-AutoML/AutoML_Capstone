"""
Tests for the quantization module.
"""
import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from src.pipeline.quantization import (
    imx_quantization,
    fp16_quantization,
    onnx_quantization,
    quantize_model
)


@pytest.fixture
def mock_model():
    """Mock YOLO model for testing."""
    model = Mock()
    model.export.return_value = "/path/to/exported/model"
    model.save = Mock()
    model.half.return_value = model
    model.model = Mock()
    model.model.half.return_value = Mock()
    return model


@pytest.fixture
def sample_quantize_config(tmp_path):
    """Sample quantization configuration."""
    return {
        "output_dir": str(tmp_path / "quantized"),
        "quantization_method": "IMX",
        "labeled_images_path": str(tmp_path / "images"),
        "calibration_samples": 100,
        "quantize_yaml_path": str(tmp_path / "quantize.yaml")
    }


def test_imx_quantization_non_linux():
    """Test IMX quantization on non-Linux platforms."""
    with patch('platform.system', return_value='Windows'):
        result = imx_quantization(Mock(), "/output/path", "quantize.yaml")
        assert result is None


@patch('platform.system', return_value='Linux')
@patch('torch.cuda.is_available', return_value=True)
def test_imx_quantization_success(mock_cuda, mock_platform, mock_model, tmp_path):
    """Test successful IMX quantization on Linux."""
    output_path = tmp_path / "quantized_model"
    exported_path = tmp_path / "exported_model"

    # Create the exported file
    exported_path.touch()
    mock_model.export.return_value = str(exported_path)

    with patch('shutil.move') as mock_move:
        result = imx_quantization(mock_model, str(output_path), "quantize.yaml")

        mock_model.export.assert_called_once_with(format="imx", data="quantize.yaml", device=0)
        mock_move.assert_called_once()
        assert result == str(output_path)


def test_fp16_quantization_success(mock_model, tmp_path):
    """Test successful FP16 quantization."""
    output_path = tmp_path / "fp16_model.pt"
    result = fp16_quantization(mock_model, str(output_path))

    mock_model.save.assert_called_once_with(str(output_path))
    assert result == str(output_path)


def test_fp16_quantization_failure(mock_model, tmp_path):
    """Test FP16 quantization failure."""
    output_path = tmp_path / "fp16_model.pt"
    mock_model.save.side_effect = Exception("Save failed")

    result = fp16_quantization(mock_model, str(output_path))
    assert result is None


@patch('subprocess.run')
@patch('src.pipeline.quantization.quantize_dynamic')
def test_onnx_quantization_success(mock_quantize, mock_subprocess, mock_model, tmp_path):
    """Test successful ONNX quantization."""
    output_path = tmp_path / "quantized.onnx"
    preprocessed_path = tmp_path / "preprocessed.onnx"
    onnx_path = tmp_path / "model.onnx"
    mock_model.export.return_value = str(onnx_path)
    onnx_path.touch()
    result = onnx_quantization(mock_model, str(output_path), str(preprocessed_path))

    mock_model.export.assert_called_once_with(format='onnx')
    mock_subprocess.assert_called_once()
    mock_quantize.assert_called_once()
    assert result == str(output_path)


@patch('src.pipeline.quantization.load_config')
@patch('src.pipeline.quantization.YOLO')
@patch('src.pipeline.quantization.prepare_quantization_data')
def test_quantize_model_imx(mock_prepare, mock_yolo, mock_load_config, sample_quantize_config, tmp_path):
    """Test quantize_model with IMX method."""
    mock_load_config.return_value = sample_quantize_config
    mock_model = Mock()
    mock_yolo.return_value = mock_model

    # Create quantize.yaml file
    yaml_path = Path(sample_quantize_config["quantize_yaml_path"])
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.touch()

    with patch('src.pipeline.quantization.imx_quantization') as mock_imx:
        mock_imx.return_value = "quantized_model_path"

        result = quantize_model("model.pt", "config.json")

        mock_prepare.assert_called_once()
        mock_imx.assert_called_once()
        assert result == "quantized_model_path"


def test_quantize_model_unsupported_method(tmp_path):
    """Test quantize_model with unsupported method."""
    config = {"quantization_method": "UNSUPPORTED", "output_dir": str(tmp_path)}

    with patch('src.pipeline.quantization.load_config', return_value=config):
        with patch('src.pipeline.quantization.YOLO'):
            with pytest.raises(ValueError, match="Unsupported quantization method"):
                quantize_model("model.pt", "config.json")


def test_quantize_model_missing_yaml():
    """Test quantize_model when quantize.yaml is missing."""
    config = {
        "quantization_method": "IMX",
        "output_dir": "/tmp",
        "labeled_images_path": "/tmp/images",
        "quantize_yaml_path": "/nonexistent/quantize.yaml"
    }

    with patch('src.pipeline.quantization.load_config', return_value=config):
        with patch('src.pipeline.quantization.YOLO'):
            with patch('src.pipeline.quantization.prepare_quantization_data'):
                with pytest.raises(ValueError, match="quantize_yaml file does not exist"):
                    quantize_model("model.pt", "config.json")
