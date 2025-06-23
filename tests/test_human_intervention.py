"""
Tests for the human_intervention module.
"""

import os
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
    _convert_bbox_from_percent,
    _generate_ls_tasks,
    _update_processed_files_status,
    _ensure_label_studio_running,
    _find_or_create_project,
    _configure_interface,
    setup_label_studio,
    import_tasks_to_project,
    export_versioned_results,
    transform_reviewed_results_to_labeled,
    _transform_ls_result_to_original_format,
    _extract_confidence_from_results,
    run_human_review
)


@pytest.fixture
def tmp_path():
    """Custom tmp_path fixture that doesn't rely on system temp directories."""
    path = Path(os.getcwd()) / "test_temp"
    if not path.exists():
        path.mkdir(parents=True)
    yield path
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(autouse=True)
def mock_global_directories():
    """Mock the global directory paths to prevent FileNotFoundError during import."""
    with patch("src.pipeline.human_intervention.label_studio_dir") as mock_ls_dir, \
         patch("src.pipeline.human_intervention.mismatch_pending_dir") as mock_pending_dir, \
         patch("src.pipeline.human_intervention.reviewed_dir") as mock_reviewed_dir, \
         patch("src.pipeline.human_intervention.image_dir") as mock_image_dir, \
         patch("src.pipeline.human_intervention.output_dir") as mock_output_dir, \
         patch("src.pipeline.human_intervention.labeled_dir") as mock_labeled_dir:

        # Configure the mock paths
        mock_ls_dir.mkdir.return_value = None
        mock_pending_dir.mkdir.return_value = None
        mock_reviewed_dir.mkdir.return_value = None
        mock_image_dir.mkdir.return_value = None
        mock_output_dir.mkdir.return_value = None
        mock_labeled_dir.mkdir.return_value = None
        yield


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    # Use absolute paths to avoid Windows path issues
    image_dir = tmp_path / "images"
    json_dir = tmp_path / "json"
    output_dir = tmp_path / "output"
    labeled_dir = tmp_path / "labeled"

    # Create directories with parents=True
    image_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    labeled_dir.mkdir(parents=True, exist_ok=True)

    # Create test images - ensure the files exist by opening and writing
    for i in range(3):
        img_jpg = image_dir / f"image_{i}.jpg"
        img_png = image_dir / f"image_{i+3}.png"

        with open(img_jpg, "wb") as f:
            f.write(b"test image content")
        with open(img_png, "wb") as f:
            f.write(b"test image content")

    # Create test JSON files
    for i in range(5):
        json_content = {
            "predictions": [
                {"bbox": [10, 20, 100, 200], "class": "FireBSI", "confidence": 0.85}
            ]
        }
        json_file = json_dir / f"image_{i}.json"
        with open(json_file, "w") as f:
            json.dump(json_content, f)

    yield image_dir, json_dir, output_dir, labeled_dir


def test_find_image_path(temp_dirs):
    """Test finding image path by stem."""
    image_dir, _, _, _ = temp_dirs

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
    _, json_dir, _, _ = temp_dirs

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
    _, json_dir, _, _ = temp_dirs

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
    test_dir.mkdir(parents=True, exist_ok=True)

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

    assert result["x"] == 5.0
    assert result["y"] == 5.0
    assert result["width"] == 25.0
    assert result["height"] == 15.0


def test_convert_bbox_from_percent():
    """Test converting bounding box from percent format back to pixels."""
    # Test with 100x100 image
    bbox_dict = {"x": 10.0, "y": 20.0, "width": 50.0, "height": 60.0}
    result = _convert_bbox_from_percent(bbox_dict, 100, 100)

    assert result == [10.0, 20.0, 60.0, 80.0]

    # Test with different image dimensions
    bbox_dict = {"x": 5.0, "y": 5.0, "width": 25.0, "height": 15.0}
    result = _convert_bbox_from_percent(bbox_dict, 200, 400)

    assert result == [10.0, 20.0, 60.0, 80.0]


@patch("PIL.Image.open")
@patch("base64.b64encode")
def test_generate_ls_tasks(mock_b64encode, mock_image_open, temp_dirs):
    """Test generating Label Studio tasks from JSON files."""
    image_dir, json_dir, output_dir, _ = temp_dirs

    # Mock Image.open to return consistent dimensions
    mock_img = MagicMock()
    mock_img.size = (100, 100)
    mock_image_open.return_value.__enter__.return_value = mock_img

    # Mock base64 encoding
    mock_b64encode.return_value.decode.return_value = "base64encodedstring"

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
        assert "filename" in task["data"]
        assert "import_timestamp" in task["data"]
        assert "original_filename" in task["data"]

        if "predictions" in task:
            assert isinstance(task["predictions"], list)
            prediction = task["predictions"][0]
            assert "result" in prediction
            assert "model_version" in prediction

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
    _, json_dir, output_dir, _ = temp_dirs

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
    _, json_dir, _, _ = temp_dirs

    # List all JSON files
    files_to_update = list(json_dir.glob("*.json"))

    # Update status
    _update_processed_files_status(files_to_update)

    # Verify all files have status = 1
    for file_path in files_to_update:
        with open(file_path, "r") as f:
            data = json.load(f)
            assert data["label_status"] == 1


@patch("subprocess.Popen")
@patch("requests.get")
def test_ensure_label_studio_running(mock_get, mock_popen):
    """Test checking if Label Studio is running."""
    # Mock successful response (already running)
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response

    # Should return True when Label Studio is running
    result = _ensure_label_studio_running()

    assert result is True
    mock_get.assert_called_once()
    mock_popen.assert_not_called()


@patch("subprocess.Popen")  
@patch("requests.get")
def test_ensure_label_studio_running_needs_start(mock_get, mock_popen):
    """Test starting Label Studio when it's not running."""
    # Mock failed first request, then successful health check
    mock_get.side_effect = [
        Exception("Connection failed"),
        Mock(status_code=200)
    ]

    result = _ensure_label_studio_running()

    assert result is True
    mock_popen.assert_called_once()
    assert mock_get.call_count >= 2


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


@patch("requests.post")
def test_import_tasks_to_project(mock_post, temp_dirs):
    """Test importing tasks to Label Studio project."""
    _, _, output_dir, _ = temp_dirs

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
def test_setup_label_studio(mock_interface, mock_project):
    """Test setting up Label Studio project."""
    mock_project.return_value = 123
    mock_interface.return_value = True

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = setup_label_studio("Test Project", "/output/dir")

    assert result == {
        "project_id": 123,
        "storage_id": None,
        "project_url": "http://localhost:8080/projects/123/data"
    }


def test_extract_confidence_from_results():
    """Test extracting confidence from Label Studio results."""
    # Test with prediction metadata
    export_result = {
        "annotations": [{
            "prediction": {
                "result": [{
                    "meta": {
                        "confidence": 0.85,
                        "confidence_flag": "high"
                    }
                }]
            }
        }]
    }

    confidence, flag = _extract_confidence_from_results(export_result, 0)
    assert confidence == 0.85
    assert flag == "high"

    # Test with no annotations
    export_result = {"annotations": []}
    confidence, flag = _extract_confidence_from_results(export_result, 0)
    assert confidence == 1.0
    assert flag == "human"


def test_transform_ls_result_to_original_format():
    """Test transforming Label Studio results back to original format."""
    export_result = {
        "id": 123,
        "data": {"original_filename": "test.json"},
        "annotations": [{
            "id": 456,
            "updated_at": "2025-01-01T00:00:00Z",
            "was_cancelled": False,
            "result": [{
                "type": "rectanglelabels",
                "value": {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 50.0,
                    "height": 60.0,
                    "rectanglelabels": ["FireBSI"]
                },
                "original_width": 100,
                "original_height": 100
            }]
        }]
    }

    result = _transform_ls_result_to_original_format(export_result, Path("/images"))

    assert result is not None
    assert result["original_filename"] == "test.json"
    assert result["data"]["label_status"] == 2
    assert len(result["data"]["predictions"]) == 1

    prediction = result["data"]["predictions"][0]
    assert prediction["bbox"] == [10.0, 20.0, 60.0, 80.0]
    assert prediction["class"] == "FireBSI"


@patch("requests.get")
@patch("pathlib.Path.glob")
@patch("json.dump")
@patch("builtins.open", create=True)
def test_export_versioned_results(mock_open, mock_json_dump, mock_glob, mock_get):
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

    # Mock file operations
    mock_path = Mock()
    mock_path.name = "image_0.json"
    mock_glob.return_value = [mock_path]

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        with patch("src.pipeline.human_intervention._update_label_status") as mock_update:
            with patch("src.pipeline.human_intervention.transform_reviewed_results_to_labeled") as mock_transform:
                mock_transform.return_value = 1
                results = export_versioned_results(
                    "123", 
                    Path("/output/dir"),
                    "v1"
                )

    assert len(results) == 2
    # Check that base64 data was removed
    assert "image" not in results[0]["data"]
    mock_transform.assert_called_once()


@patch("json.dump")
@patch("builtins.open", create=True)
@patch("pathlib.Path.unlink")
@patch("pathlib.Path.exists")
def test_transform_reviewed_results_to_labeled(mock_exists, mock_unlink, mock_open, mock_json_dump, temp_dirs):
    """Test transforming reviewed results to labeled directory."""
    _, _, _, labeled_dir = temp_dirs

    # Mock export results
    export_results = [{
        "id": 123,
        "data": {"original_filename": "test.json"},
        "annotations": [{
            "id": 456,
            "updated_at": "2025-01-01T00:00:00Z",
            "was_cancelled": False,
            "result": [{
                "type": "rectanglelabels",
                "value": {
                    "x": 10.0,
                    "y": 20.0,
                    "width": 50.0,
                    "height": 60.0,
                    "rectanglelabels": ["FireBSI"]
                },
                "original_width": 100,
                "original_height": 100
            }]
        }]
    }]

    # Mock file operations
    mock_exists.return_value = False

    with patch("src.pipeline.human_intervention.mismatch_pending_dir") as mock_pending:
        mock_pending_file = Mock()
        mock_pending_file.exists.return_value = True
        mock_pending.__truediv__.return_value = mock_pending_file

        count = transform_reviewed_results_to_labeled(
            export_results, 
            labeled_dir, 
            Path("/images")
        )

    assert count == 1
    mock_json_dump.assert_called_once()
    mock_pending_file.unlink.assert_called_once()


@patch("src.pipeline.human_intervention.setup_label_studio")
@patch("src.pipeline.human_intervention._ensure_label_studio_running")
@patch("src.pipeline.human_intervention._ensure_directories")
@patch("src.pipeline.human_intervention._initialize_json_files")
@patch("src.pipeline.human_intervention._generate_ls_tasks")
@patch("src.pipeline.human_intervention.import_tasks_to_project")
@patch("src.pipeline.human_intervention._update_processed_files_status")
@patch("pathlib.Path.glob")
def test_run_human_review(
    mock_glob, mock_update_status, mock_import, mock_generate, mock_initialize, 
    mock_ensure_dirs, mock_ensure_running, mock_setup
):
    """Test running the complete human review workflow."""
    # Setup mocks
    mock_ensure_running.return_value = True
    mock_setup.return_value = {
        "project_id": 123,
        "storage_id": None,
        "project_url": "http://localhost:8080/projects/123/data"
    }
    mock_generate.return_value = (Path("/tmp/tasks.json"), ["file1.json", "file2.json"])
    mock_import.return_value = True
    mock_glob.return_value = ["file1.json", "file2.json"]

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = run_human_review("Test Project", export_results_flag=False)

    assert result == {
        "project_id": 123,
        "storage_id": None,
        "project_url": "http://localhost:8080/projects/123/data"
    }
    mock_ensure_dirs.assert_called_once()
    mock_update_status.assert_called_once()


@patch("src.pipeline.human_intervention.setup_label_studio")
@patch("src.pipeline.human_intervention._ensure_label_studio_running")
@patch("src.pipeline.human_intervention._ensure_directories")
@patch("src.pipeline.human_intervention._initialize_json_files")
def test_run_human_review_setup_failure(mock_initialize, mock_ensure_dirs, mock_ensure_running, mock_setup):
    """Test run_human_review when setup fails."""
    mock_ensure_running.return_value = True
    mock_setup.return_value = {}

    # Should return empty dict when setup fails
    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = run_human_review("Test Project", export_results_flag=False)

    assert result == {}


@patch("src.pipeline.human_intervention._ensure_label_studio_running")
@patch("src.pipeline.human_intervention._ensure_directories")
@patch("src.pipeline.human_intervention._initialize_json_files")
def test_run_human_review_no_label_studio(mock_initialize, mock_ensure_dirs, mock_ensure_running):
    """Test running workflow when Label Studio isn't running."""
    mock_ensure_running.return_value = False

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = run_human_review("Test Project", export_results_flag=False)

    assert result == {}


@patch("src.pipeline.human_intervention.export_versioned_results")
@patch("src.pipeline.human_intervention.setup_label_studio")
@patch("src.pipeline.human_intervention._ensure_label_studio_running")
@patch("src.pipeline.human_intervention._ensure_directories")
@patch("src.pipeline.human_intervention._initialize_json_files")
@patch("src.pipeline.human_intervention._generate_ls_tasks")
@patch("src.pipeline.human_intervention.import_tasks_to_project")
@patch("builtins.input")
@patch("pathlib.Path.glob")
def test_run_human_review_with_export(
    mock_glob, mock_input, mock_import, mock_generate, mock_initialize, 
    mock_ensure_dirs, mock_ensure_running, mock_setup, mock_export
):
    """Test running workflow with export enabled."""
    # Setup mocks
    mock_ensure_running.return_value = True
    mock_setup.return_value = {
        "project_id": 123,
        "storage_id": None,
        "project_url": "http://localhost:8080/projects/123/data"
    }
    mock_generate.return_value = (None, [])
    mock_export.return_value = [{"id": 1, "data": {}}]
    mock_input.return_value = ""
    mock_glob.return_value = []

    with patch.dict("os.environ", {"LABEL_STUDIO_API_KEY": "test_key"}):
        result = run_human_review("Test Project", export_results_flag=True)

    assert isinstance(result, list)
    mock_export.assert_called_once()
