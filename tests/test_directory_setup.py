"""
Tests for the directory_setup module.
"""
import os
from unittest.mock import patch
from src.directory_setup import create_automl_workspace


def test_create_automl_workspace_default_path(tmp_path):
    """Test creating automl_workspace with default path."""
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        create_automl_workspace()

        # Verify workspace was created
        workspace_dir = tmp_path / "automl_workspace"
        assert workspace_dir.exists()
        assert workspace_dir.is_dir()

        # Verify subdirectories exist
        expected_dirs = [
            "data_pipeline/input",
            "data_pipeline/prelabeled",
            "data_pipeline/labeled",
            "data_pipeline/augmented",
            "data_pipeline/training",
            "data_pipeline/distillation",
            "data_pipeline/quantization",
            "data_pipeline/label_studio/pending",
            "data_pipeline/label_studio/tasks",
            "data_pipeline/label_studio/results",
            "model_registry/model",
            "model_registry/distilled",
            "model_registry/quantized",
            "master_dataset",
            "config"
        ]

        for subdir in expected_dirs:
            dir_path = workspace_dir / subdir
            assert dir_path.exists(), f"Directory {subdir} was not created"
            assert dir_path.is_dir(), f"{subdir} is not a directory"

    finally:
        os.chdir(original_cwd)


def test_create_automl_workspace_custom_path(tmp_path):
    """Test creating automl_workspace with custom base path."""
    custom_base = tmp_path / "custom_location"
    custom_base.mkdir()

    create_automl_workspace(str(custom_base))

    # Verify workspace was created in custom location
    workspace_dir = custom_base / "automl_workspace"
    assert workspace_dir.exists()

    # Verify key directories
    assert (workspace_dir / "data_pipeline" / "input").exists()
    assert (workspace_dir / "model_registry" / "model").exists()
    assert (workspace_dir / "config").exists()


def test_create_automl_workspace_already_exists(tmp_path):
    """Test that function handles existing directories gracefully."""
    create_automl_workspace(str(tmp_path))

    # Create test file
    test_file = tmp_path / "automl_workspace" / "config" / "test.txt"
    test_file.write_text("test content")

    create_automl_workspace(str(tmp_path))

    # Verify the test file still exists
    assert test_file.exists()
    assert test_file.read_text() == "test content"


def test_create_automl_workspace_permission_error(tmp_path):
    """Test handling of permission errors during directory creation."""
    with patch('os.makedirs') as mock_makedirs:
        mock_makedirs.side_effect = PermissionError("Permission denied")

        with patch('builtins.print') as mock_print:
            create_automl_workspace(str(tmp_path))

            assert mock_print.called
            error_calls = [call for call in mock_print.call_args_list
                           if "Error creating directory" in str(call)]
            assert len(error_calls) > 0


def test_create_automl_workspace_nested_structure(tmp_path):
    """Test that nested directory structure is created correctly."""
    create_automl_workspace(str(tmp_path))
    workspace_dir = tmp_path / "automl_workspace"

    # Test deep nesting
    label_studio_pending = workspace_dir / "data_pipeline" / "label_studio" / "pending"
    assert label_studio_pending.exists()
    assert label_studio_pending.is_dir()

    # Test parent directories exist
    assert (workspace_dir / "data_pipeline").exists()
    assert (workspace_dir / "data_pipeline" / "label_studio").exists()


@patch('os.path.exists')
@patch('os.makedirs')
def test_create_automl_workspace_makedirs_called_correctly(mock_makedirs, mock_exists):
    """Test that os.makedirs is called with correct paths."""
    mock_exists.return_value = False
    create_automl_workspace("/test/path")

    # Check that makedirs was called for each expected subdirectory
    made_dirs = [call[0][0] for call in mock_makedirs.call_args_list]
    assert any("data_pipeline/input" in path for path in made_dirs)
    assert any("model_registry/model" in path for path in made_dirs)
    assert any("config" in path for path in made_dirs)
