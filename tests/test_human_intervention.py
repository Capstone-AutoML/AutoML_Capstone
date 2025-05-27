"""
Tests for the human_intervention module.
"""

import pytest
import json
from pathlib import Path
import shutil
from unittest.mock import patch, Mock, MagicMock

from src.pipeline.human_intervention import (
    _find_image_path,
    _update_label_status,
    _initialize_json_files,
    _convert_bbox_to_percent,
    _generate_ls_tasks,
    _update_processed_files_status,
    _ensure_label_studio_running,
    _find_or_create_project,
    _configure_interface,
    _connect_local_storage,
    setup_label_studio,
    import_tasks_to_project,
    export_versioned_results,
    run_human_review
)


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "json"
    output_dir = tmp_path / "output"

    # Create directories
    image_dir.mkdir()
    json_dir.mkdir()
    output_dir.mkdir()

    # Create test images
    for i in range(3):
        (image_dir / f"image_{i}.jpg").touch()
        (image_dir / f"image_{i+3}.png").touch()

    # Create test JSON files
    for i in range(5):
        json_content = {
            "predictions": [
                {"bbox": [10, 20, 100, 200], "class": "FireBSI"}
            ]
        }
        with open(json_dir / f"image_{i}.json", "w") as f:
            json.dump(json_content, f)

    yield image_dir, json_dir, output_dir

    # Cleanup
    shutil.rmtree(tmp_path)


def test_find_image_path(temp_dirs):
    """Test finding image path by stem."""
    image_dir, _, _ = temp_dirs

    # Should find existing image
    path = _find_image_path("image_0", image_dir)
    assert path == image_dir / "image_0.jpg"

    path = _find_image_path("image_3", image_dir)
    assert path == image_dir / "image_3.png"

    # Should return None for non-existent image
    path = _find_image_path("nonexistent", image_dir)
    assert path is None


def test_update_label_status(temp_dirs):
    """Test updating label status in JSON file."""
    _, json_dir, _ = temp_dirs

    # Create a test JSON file
    file_path = json_dir / "test.json"
    with open(file_path, "w") as f:
        json.dump({"predictions": []}, f)

    # Update status
    result = _update_label_status(file_path, 1)
    assert result is True

    # Verify status was updated
    with open(file_path, "r") as f:
        data = json.load(f)
        assert data["label_status"] == 1


def test_update_label_status_file_not_found(tmp_path):
    """Test updating label status for non-existent file."""
    non_existent_file = tmp_path / "does_not_exist.json"

    # Return False when file does not exist
    result = _update_label_status(non_existent_file, 1)
    assert result is False


def test_initialize_json_files(temp_dirs):
    """Test initializing label_status field in JSON files."""
    _, json_dir, _ = temp_dirs

    # Initialize files
    count = _initialize_json_files(json_dir)

    # Should have initialized 5 files
    assert count == 5

    # Verify each file has label_status = 0
    for json_file in json_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)
            assert "label_status" in data
            assert data["label_status"] == 0

    # Second call should return 0 since all files are already initialized
    count = _initialize_json_files(json_dir)
    assert count == 0


def test_initialize_json_files_empty_dir(tmp_path):
    """Test initializing an empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Should return 0 with no JSON files
    count = _initialize_json_files(empty_dir)
    assert count == 0


def test_initialize_json_files_invalid_json(tmp_path):
    """Test initializing files with invalid JSON content."""
    test_dir = tmp_path / "invalid"
    test_dir.mkdir()

    # Create file with invalid JSON
    invalid_file = test_dir / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("This is not valid JSON")

    # Should handle the exception without crashing
    count = _initialize_json_files(test_dir)
    assert count == 0


def test_convert_bbox_to_percent():
    """Test converting bounding box to percent format."""
    # Test with 100x100 image
    bbox = [10, 20, 60, 80]
    result = _convert_bbox_to_percent(bbox, 100, 100)

    assert result["x"] == 10.0
    assert result["y"] == 20.0
    assert result["width"] == 50.0
    assert result["height"] == 60.0

    # Test with different image dimensions
    bbox = [10, 20, 60, 80]
    result = _convert_bbox_to_percent(bbox, 200, 400)

    assert result["x"] == 5.0  # 10/200 * 100
    assert result["y"] == 5.0  # 20/400 * 100
    assert result["width"] == 25.0  # (60-10)/200 * 100
    assert result["height"] == 15.0  # (80-20)/400 * 100


@patch("PIL.Image.open")
def test_generate_ls_tasks(mock_image_open, temp_dirs):
    """Test generating Label Studio tasks from JSON files."""
    image_dir, json_dir, output_dir = temp_dirs

    # Mock Image.open to return consistent dimensions
    mock_img = MagicMock()
    mock_img.size = (100, 100)
    mock_image_open.return_value.__enter__.return_value = mock_img

    # Test with files that have no label_status yet
    versioned_file, processed_files = _generate_ls_tasks(json_dir, image_dir, output_dir)

    assert versioned_file is not None
    assert versioned_file.exists()
    assert len(processed_files) > 0

    # Verify the output file structure
    with open(versioned_file, "r") as f:
        tasks = json.load(f)
        assert isinstance(tasks, list)
        assert len(tasks) > 0

        # Check task structure
        task = tasks[0]
        assert "data" in task
        assert "image" in task["data"]
        assert task["data"]["image"].startswith("data:image/")

        if "predictions" in task:
            assert isinstance(task["predictions"], list)
            prediction = task["predictions"][0]
            assert "result" in prediction

    # Test with files that have been imported (status = 1)
    for file in json_dir.glob("*.json"):
        _update_label_status(file, 1)

    versioned_file, processed_files = _generate_ls_tasks(json_dir, image_dir, output_dir)

    # Should return None since no new files to process
    assert versioned_file is None
    assert len(processed_files) == 0


@patch("PIL.Image.open")
def test_generate_ls_tasks_empty_dir(mock_image_open, tmp_path):
    """Test generating tasks from empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Should return None and empty list for no files
    versioned_file, processed_files = _generate_ls_tasks(empty_dir, tmp_path, output_dir)
    assert versioned_file is None
    assert processed_files == []


@patch("PIL.Image.open")
def test_generate_ls_tasks_missing_images(mock_image_open, temp_dirs):
    """Test handling JSON files with missing corresponding images."""
    _, json_dir, output_dir = temp_dirs

    # Create a non-existent image directory
    missing_img_dir = Path("/non/existent/path")

    # Should either return None or empty processed files list
    versioned_file, processed_files = _generate_ls_tasks(
        json_dir,
        missing_img_dir,
        output_dir
    )
    assert versioned_file is None or len(processed_files) == 0


def test_update_processed_files_status(temp_dirs):
    """Test updating status of processed files."""
    _, json_dir, _ = temp_dirs

    # List all JSON files
    files_to_update = list(json_dir.glob("*.json"))

    # Update status
    _update_processed_files_status(files_to_update)

    # Verify all files have status = 1
    for file_path in files_to_update:
        with open(file_path, "r") as f:
            data = json.load(f)
            assert data["label_status"] == 1


@patch("requests.get")
def test_ensure_label_studio_running(mock_get):
    """Test checking if Label Studio is running."""
    # Mock successful response
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Should return True when Label Studio is running
    result = _ensure_label_studio_running()

    assert result is True
    mock_get.assert_called_once()


@patch("requests.get")
@patch("requests.post")
def test_find_or_create_project(mock_post, mock_get):
    """Test finding or creating Label Studio project."""
    # Mock responses
    mock_get_response = Mock()
    mock_get_response.json.return_value = {"results": [
        {"id": 123, "title": "Existing Project"}
    ]}
    mock_get_response.status_code = 200
    mock_get.return_value = mock_get_response

    mock_post_response = Mock()
    mock_post_response.json.return_value = {"id": 456}
    mock_post_response.status_code = 201
    mock_post.return_value = mock_post_response

    # Test finding existing project
    project_id = _find_or_create_project(
        "http://localhost:8080",
        {"Authorization": "Token abc123"},
        "Existing Project"
    )
    assert project_id == 123

    # Test creating new project
    project_id = _find_or_create_project(
        "http://localhost:8080", 
        {"Authorization": "Token abc123"}, 
        "New Project"
    )
    assert project_id == 456
    mock_post.assert_called_once()


@patch("requests.patch")
def test_configure_interface(mock_patch):
    """Test configuring Label Studio interface."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_patch.return_value = mock_response

    result = _configure_interface(
        "http://localhost:8080",
        {"Authorization": "Token abc123"},
        123
    )

    assert result is True
    mock_patch.assert_called_once()


@patch("requests.get")
@patch("requests.post")
def test_connect_local_storage(mock_post, mock_get):
    """Test connecting Label Studio to local storage."""
    # Mock responses
    mock_get_response = Mock()
    mock_get_response.json.return_value = []  # No existing storage
    mock_get_response.status_code = 200
    mock_get.return_value = mock_get_response

    mock_post_response = Mock()
    mock_post_response.json.return_value = {"id": 789}
    mock_post_response.status_code = 201
    mock_post.return_value = mock_post_response

    # Test creating new storage
    result = _connect_local_storage(
        "http://localhost:8080",
        {"Authorization": "Token abc123"},
        123,
        "Test Project",
        "/path/to/data"
    )

    assert result == 789
    assert mock_post.call_count == 2


@patch("requests.post")
@patch("requests.get")
def test_import_tasks_to_project(mock_get, mock_post, temp_dirs):
    """Test importing tasks to Label Studio project."""
    _, _, output_dir = temp_dirs

    # Create test tasks file
    tasks_file = output_dir / "tasks.json"
    with open(tasks_file, "w") as f:
        json.dump([{"data": {"image": "test.jpg"}}], f)

    # Mock response
    mock_response = Mock()
    mock_response.status_code = 201
    mock_post.return_value = mock_response

    result = import_tasks_to_project(
        "http://localhost:8080",
        {"Authorization": "Token abc123"},
        123,
        tasks_file
    )

    assert result is True
    mock_post.assert_called_once()


@patch("src.pipeline.human_intervention._find_or_create_project")
@patch("src.pipeline.human_intervention._configure_interface")
@patch("src.pipeline.human_intervention._connect_local_storage")
def test_setup_label_studio(mock_storage, mock_interface, mock_project):
    """Test setting up Label Studio project."""
    mock_project.return_value = 123
    mock_interface.return_value = True
    mock_storage.return_value = 789

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = setup_label_studio("Test Project", "/output/dir")

    assert result == {
        "project_id": 123,
        "storage_id": 789,
        "project_url": "http://localhost:8080/projects/123/data"
    }


@patch("requests.get")
def test_export_versioned_results(mock_get):
    """Test exporting versioned results from Label Studio."""
    # Mock responses
    tasks_response = Mock()
    tasks_response.json.return_value = [
        {
            "id": 1,
            "annotations": [{"result": []}],
            "data": {"original_filename": "image_0.json"}
        },
        {
            "id": 2,
            "annotations": [],
            "data": {"original_filename": "image_1.json"}
        }
    ]
    tasks_response.status_code = 200

    export_response = Mock()
    export_response.json.return_value = [
        {
            "id": 1,
            "data": {"image": "base64data", "filename": "image_0.jpg"}
        },
        {
            "id": 2,
            "data": {"image": "base64data", "filename": "image_1.jpg"}
        }
    ]
    export_response.status_code = 200

    mock_get.side_effect = [tasks_response, export_response]

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        with patch("builtins.open", create=True):
            with patch("json.dump"):
                with patch("pathlib.Path.glob") as mock_glob:
                    # Mock glob to return a path
                    mock_path = Mock()
                    mock_path.name = "image_0.json"
                    mock_glob.return_value = [mock_path]

                    # Test export
                    results = export_versioned_results(
                        "123", 
                        Path("/output/dir"),
                        "v1"
                    )

    assert len(results) == 2
    # Check that base64 data was removed
    assert "image" not in results[0]["data"]


@patch("requests.get")
def test_export_versioned_results_missing_api_key(mock_get):
    """Test export with missing API key."""
    # Should return empty dict when API key is missing
    with patch.dict("os.environ", {}, clear=True):
        result = export_versioned_results("123", Path("/tmp"))
    assert result == {}


@patch("src.pipeline.human_intervention.setup_label_studio")
@patch("src.pipeline.human_intervention._ensure_label_studio_running")
@patch("src.pipeline.human_intervention._initialize_json_files")
@patch("src.pipeline.human_intervention._generate_ls_tasks")
@patch("src.pipeline.human_intervention.import_tasks_to_project")
@patch("src.pipeline.human_intervention._update_processed_files_status")
def test_run_human_review(
    mock_update_status, mock_import, mock_generate, mock_initialize, 
    mock_ensure_running, mock_setup
):
    """Test running the complete human review workflow."""
    # Setup mocks
    mock_ensure_running.return_value = True
    mock_setup.return_value = {
        "project_id": 123,
        "storage_id": 789,
        "project_url": "http://localhost:8080/projects/123/data"
    }
    mock_generate.return_value = (Path("/tmp/tasks.json"), ["file1.json", "file2.json"])
    mock_import.return_value = True

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        with patch("pathlib.Path.glob") as mock_glob:
            mock_glob.return_value = ["file1.json", "file2.json"]
            result = run_human_review("Test Project", export_results_flag=False)

    assert result == {
        "project_id": 123,
        "storage_id": 789,
        "project_url": "http://localhost:8080/projects/123/data"
    }
    mock_update_status.assert_called_once()


@patch("src.pipeline.human_intervention.setup_label_studio")
@patch("src.pipeline.human_intervention._ensure_label_studio_running")
@patch("src.pipeline.human_intervention._initialize_json_files")
def test_run_human_review_setup_failure(mock_initialize, mock_ensure_running, mock_setup):
    """Test run_human_review when setup fails."""
    mock_ensure_running.return_value = True
    mock_setup.return_value = {}

    # Should return empty dict when setup fails
    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = run_human_review("Test Project", export_results_flag=False)

    assert result == {}


@patch("src.pipeline.human_intervention.setup_label_studio")
@patch("src.pipeline.human_intervention._ensure_label_studio_running")
@patch("src.pipeline.human_intervention._initialize_json_files")
@patch("src.pipeline.human_intervention._generate_ls_tasks")
@patch("src.pipeline.human_intervention.import_tasks_to_project")
def test_run_human_review_import_failure(
    mock_import, mock_generate, mock_initialize, mock_ensure_running, mock_setup
):
    """Test run_human_review when import fails."""
    mock_ensure_running.return_value = True
    mock_setup.return_value = {
        "project_id": 123,
        "storage_id": 789,
        "project_url": "http://localhost:8080/projects/123/data"
    }
    mock_generate.return_value = (Path("/tmp/tasks.json"), ["file1.json", "file2.json"])
    mock_import.return_value = False

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = run_human_review("Test Project", export_results_flag=False)

    # Should still return project details even if import fails
    assert "project_url" in result


@patch("src.pipeline.human_intervention._ensure_label_studio_running")
def test_run_human_review_no_label_studio(mock_ensure_running):
    """Test running workflow when Label Studio isn't running."""
    mock_ensure_running.return_value = False

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = run_human_review("Test Project", export_results_flag=False)

    assert result == {}
