"""
Unit tests for the train.py module.
"""

import pytest
import json

from src.pipeline.train import load_train_config, find_latest_model


def test_load_valid_config(tmp_path):
    """
    Test that a valid training config JSON file loads correctly.
    """
    config_path = tmp_path / "config.json"
    config_data = {
        "training_config": {"epochs": 5, "lr0": 0.01},
        "data_yaml_path": "mock_io/model_registry/model/data.yaml",
        "initial_model_path": "mock_io/model_registry/model/nano_trained_model.pt"
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f)

    loaded = load_train_config(str(config_path))
    assert loaded["training_config"]["epochs"] == 5
    assert loaded["data_yaml_path"] == "mock_io/model_registry/model/data.yaml"
    assert loaded["initial_model_path"].endswith("nano_trained_model.pt")


def test_missing_file_raises_error():
    """
    Test that loading a non-existent config file raises FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        load_train_config("non_existent_file.json")


def test_missing_required_keys(tmp_path):
    """
    Test that a config file missing required keys raises AssertionError.
    """
    bad_config = {
        "training_config": {"epochs": 5}
        # Missing "data_yaml_path"
    }
    config_path = tmp_path / "bad_config.json"
    with open(config_path, "w") as f:
        json.dump(bad_config, f)

    with pytest.raises(AssertionError):
        load_train_config(str(config_path))


def test_returns_latest_model(tmp_path):
    """
    Test that the filename-encoded latest *_updated_yolo.pt model is selected.
    """
    (tmp_path / "2024-01-01_10-00-00_updated_yolo.pt").touch()
    (tmp_path / "2024-01-01_15-00-00_updated_yolo.pt").touch()
    (tmp_path / "2024-01-01_12-00-00_updated_yolo.pt").touch()

    fallback = tmp_path / "fallback.pt"
    latest = find_latest_model(str(tmp_path), str(fallback))

    assert latest.endswith("2024-01-01_15-00-00_updated_yolo.pt")


def test_returns_fallback_if_no_model(tmp_path):
    """
    Test that find_latest_model returns the fallback if no valid model files are found.
    """
    fallback = tmp_path / "fallback.pt"
    fallback.touch()

    result = find_latest_model(str(tmp_path), str(fallback))
    assert str(result) == str(fallback)


def test_model_pattern_only_matches_updated(tmp_path):
    """
    Test that find_latest_model only considers *_updated_yolo.pt files.
    """
    (tmp_path / "model_backup.pt").touch()
    (tmp_path / "not_a_model.pt").touch()

    fallback = tmp_path / "fallback.pt"
    fallback.touch()

    result = find_latest_model(str(tmp_path), str(fallback))
    assert str(result) == str(fallback)
