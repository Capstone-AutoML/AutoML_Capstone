"""
Script for human review of the unmatched YOLO and SAM results.
"""
import os
import json
import time
import sys
import base64
import requests
import subprocess
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv


mismatch_dir = Path("mock_io/data/mismatched")
mismatch_pending_dir = mismatch_dir / "pending"
reviewed_dir = mismatch_dir / "reviewed_results"
image_dir = Path("mock_io/data/raw/images")
output_dir = Path("mock_io/data/ls_tasks")
labeled_dir = Path("mock_io/data/labeled")
mismatch_dir.mkdir(exist_ok=True)
mismatch_pending_dir.mkdir(exist_ok=True)
reviewed_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)
labeled_dir.mkdir(exist_ok=True)


def _find_image_path(stem: str, image_dir: Path) -> Path:
    """
    Finds image file in image_dir by stem.

    Args:
        stem (str): The filename without extension.
        image_dir (Path): Directory where images are stored.

    Returns:
        Path: Path to the image file if found, else None.
    """
    for ext in [".jpg", ".jpeg", ".png"]:
        img_path = image_dir / f"{stem}{ext}"
        if img_path.is_file():
            return img_path
    return None


def _update_label_status(file_path: Path, status: int):
    """
    Update the label_status field in a JSON file.

    Args:
        file_path (Path): Path to the JSON file.
        status (int): Label status value.
            - 0: Unimported
            - 1: Imported & Unlabeled
            - 2: Imported & Labeled
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        data["label_status"] = status
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"[Error] Failed to update label status for {file_path.name}: {e}")
        return False


def _initialize_json_files(json_dir: Path):
    """
    Initialize all JSON files label_status=0 if they don't have one yet.

    Args:
        json_dir (Path): Directory containing JSON files to initialize

    Returns:
        int: Number of files initialized
    """
    count = 0
    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            if "label_status" not in data:
                data["label_status"] = 0
                with open(json_file, "w") as f:
                    json.dump(data, f, indent=2)
                count += 1
        except Exception as e:
            print(f"[Error] Failed to initialize status for {json_file.name}: {e}")
    if count > 0:
        print(f"[✓] Initialized {count} files with status = 0")
    return count


def _convert_bbox_to_percent(bbox, img_width, img_height):
    """
    Converts bounding box coordinates to percentage.
    This is Label Studio's expected format for bounding boxes.

    Args:
        bbox (list): Bounding box in [x1, y1, x2, y2] pixel format.
        img_width (int): Width of the image in pixels.
        img_height (int): Height of the image in pixels.

    Returns:
        dict: Bounding box with relative x, y, width, and height (0–100).
    """
    x1, y1, x2, y2 = bbox
    x = (x1 / img_width) * 100
    y = (y1 / img_height) * 100
    width = ((x2 - x1) / img_width) * 100
    height = ((y2 - y1) / img_height) * 100
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height
    }


def _convert_bbox_from_percent(bbox_dict, img_width, img_height):
    """
    Converts bounding box coordinates from percentage back to pixel format.

    Args:
        bbox_dict (dict): Bbox with x, y, width, height in percentages
        img_width (int): Width of the image in pixels
        img_height (int): Height of the image in pixels

    Returns:
        list: Bounding box in [x1, y1, x2, y2] pixel format
    """
    x_percent = bbox_dict["x"]
    y_percent = bbox_dict["y"]
    width_percent = bbox_dict["width"]
    height_percent = bbox_dict["height"]

    x1 = (x_percent / 100) * img_width
    y1 = (y_percent / 100) * img_height
    x2 = x1 + (width_percent / 100) * img_width
    y2 = y1 + (height_percent / 100) * img_height

    return [x1, y1, x2, y2]


def _generate_ls_tasks(json_dir: Path, image_dir: Path, output_dir: Path):
    """
    Converts mismatched YOLOv8 prediction JSON files into a single
    Label Studio import file for human review, using base64 encoding.

    This function only processes files with label_status = 0 (unimported).

    Args:
        json_dir (Path): Directory containing pending YOLO prediction `.json` files.
        image_dir (Path): Directory containing the corresponding image files.
        output_dir (Path): Path to save the generated `tasks.json` for LS.
    """
    # Create a version ID based on the current date and time
    version_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    tasks = []
    processed_files = []

    for json_file in json_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                original = json.load(f)

            # Skip files that are already imported (status > 0)
            label_status = original.get("label_status", 0)
            if label_status > 0:
                continue

            stem = json_file.stem
            image_path = _find_image_path(stem, image_dir)

            if not image_path:
                print(f"[Skip] Image not found for: {stem}")
                continue

            with Image.open(image_path) as img:
                width, height = img.size

            # Convert YOLOv8 predictions to Label Studio task format
            prediction_data = []
            for item in original.get("predictions", []):
                box = _convert_bbox_to_percent(item["bbox"], width, height)
                box.update({
                    "rectanglelabels": [item["class"]]
                })
                prediction_data.append({
                    "from_name": "bbox",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "value": box,
                    "meta": {
                        "confidence": item.get("confidence", 0.0),
                        "confidence_flag": item.get("confidence_flag", "N.A.")
                    }
                })

            # Encode image to base64
            with open(image_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')

            # Get format from file extension
            img_format = image_path.suffix.lstrip('.').lower()
            if not img_format:
                img_format = 'jpeg' if img_format == 'jpg' else img_format

            # Create task with base64 encoded image and metadata
            task = {
                "data": {
                    "image": f"data:image/{img_format};base64,{img_base64}",
                    "filename": image_path.name,
                    "import_timestamp": datetime.now().isoformat(),
                    "original_filename": json_file.name
                }
            }

            if prediction_data:
                task["predictions"] = [{
                    "model_version": "yolov8-prelabel-mismatch",
                    "result": prediction_data
                }]

            tasks.append(task)

            # Track this file as successfully processed
            processed_files.append(json_file)

        except Exception as e:
            print(f"[Error] Failed to process {json_file.name}: {e}")
            continue

    if not tasks:
        print("[Info] No new files to process (all files have label_status > 0)")
        return None, []

    # Save the task file with version ID
    versioned_file = output_dir / f"tasks_{version_id}.json"
    with open(versioned_file, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"[√] Exported {len(tasks)} tasks to {versioned_file}")

    return versioned_file, processed_files


def _update_processed_files_status(files_to_update: list):
    """
    Update successfully processed JSON files to label_status = 1 (imported).

    This tracks processing status in the JSON files.

    Args:
        files_to_update: List of Path objects representing JSON files to update
    """
    updated_count = 0
    for file_path in files_to_update:
        if _update_label_status(file_path, 1):
            updated_count += 1

    print(f"[✓] Imported {updated_count}/{len(files_to_update)} files")


# def _move_files_to_imported(files_to_move: list):
#     """
#     Move successfully processed JSON files
#     from 'pending' directory to 'imported' directory.

#     This helps track which files have already been imported into Label Studio
#     and prevents processing the same files twice.

#     Args:
#         files_to_move: List of Path objects representing JSON files to move
#     """
#     for file_path in files_to_move:
#         dest_path = mismatch_imported_dir / file_path.name
#         try:
#             shutil.move(str(file_path), str(dest_path))
#         except Exception as e:
#             print(f"[Error] Failed to move {file_path.name}: {e}")


def _ensure_label_studio_running():
    """
    Start Label Studio if it's not already running.

    This function checks if Label Studio is running on port 8080, and if not,
    starts it with the proper settings for accessing local files.

    Returns:
        bool: True if Label Studio is running successfully, False otherwise
    """
    # Get absolute path to data directory
    data_dir = str(Path("mock_io/data").resolve())

    # Check if Label Studio is already running
    try:
        base_url = "http://localhost:8080"
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            print("[✓] Label Studio is already running")
            return True
    except Exception:
        print("[Info] Starting Label Studio...")

    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED"] = "true"
    env["LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT"] = data_dir

    # Start Label Studio based on operating system
    if sys.platform == 'win32':  # Windows
        subprocess.Popen(["label-studio", "start"], env=env, creationflags=subprocess.CREATE_NEW_CONSOLE)
    else:  # Linux, macOS, etc.
        subprocess.Popen(["label-studio", "start"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Wait for Label Studio to start
    print("[Info] Waiting for Label Studio to start...")
    for _ in range(10):
        try:
            response = requests.get("http://localhost:8080/health", timeout=2)
            if response.status_code == 200:
                print("[✓] Label Studio started")
                return True
        except Exception:
            time.sleep(2)

    print("[Warning] Label Studio may not have started correctly")
    return False


def _find_or_create_project(base_url: str, headers: Dict[str, str],
                            project_name: str) -> Optional[int]:
    """
    Find an existing Label Studio project by name or create a new one.

    First checks if a project with the specified name exists. If found,
    returns its ID. Otherwise, creates a new project with the given name.

    Args:
        base_url: Label Studio server URL
        headers: HTTP headers including the API key
        project_name: Name of the project

    Returns:
        int: Project ID if successful, None if failed
    """
    try:
        # Get projects list
        response = requests.get(f"{base_url}/api/projects", headers=headers)
        response.raise_for_status()
        projects = response.json()["results"]

        # Look for matching project
        for project in projects:
            if isinstance(project, dict) and project.get("title") == project_name:
                print(f"[✓] Using existing project: {project_name} (ID: {project['id']})")
                return project["id"]

        print(f"[Debug] No project with title '{project_name}' found, creating new one")

    except Exception as e:
        print(f"[Error] Failed to retrieve projects: {e}")

    # Create new project
    try:
        project_data = {
            "title": project_name,
            "description": "Human review of mismatched YOLO and SAM detection results"
        }
        response = requests.post(
            f"{base_url}/api/projects",
            headers=headers,
            json=project_data
        )
        response.raise_for_status()
        project_id = response.json()["id"]
        print(f"[✓] Created new project: {project_name} (ID: {project_id})")
        return project_id

    except Exception as e:
        print(f"[Error] Failed to create project: {e}")
        return None


def _configure_interface(base_url: str, headers: Dict[str, str],
                         project_id: int) -> bool:
    """
    Configure the labeling interface for the project.

    Sets up the configuration for the Label Studio interface.

    Args:
        base_url: Label Studio server URL
        headers: HTTP headers including the API authorization token
        project_id: ID of the project to configure

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        labeling_config = """
        <View>
          <Header value="Verify bounding boxes for the labeled objects"/>
          <Image name="image" value="$image" zoom="true"/>

          <RectangleLabels name="bbox" toName="image">
            <Label value="FireBSI" background="red" hotkey="f"/>
            <Label value="SmokeBSI" background="gray" hotkey="s"/>
            <Label value="LightningBSI" background="purple" hotkey="l"/>
            <Label value="VehicleBSI" background="green" hotkey="v"/>
            <Label value="PersonBSI" background="orange" hotkey="p"/>
          </RectangleLabels>

          <TextArea name="comments" toName="image"
                   placeholder="Optional: Add notes about this image..."
                   maxSubmissions="1" editable="true" />
        </View>
        """

        response = requests.patch(
            f"{base_url}/api/projects/{project_id}",
            headers=headers,
            json={"label_config": labeling_config}
        )
        response.raise_for_status()
        print(f"[✓] Configured labeling interface")
        return True

    except requests.RequestException as e:
        print(f"[Error] Failed to configure interface: {e}")
        return False


def _connect_local_storage(base_url: str, headers: Dict[str, str],
                           project_id: int, project_name: str,
                           output_dir: str) -> Optional[int]:
    """
    Connect a Label Studio project to the local file system.

    This sets up or reuses a local storage so the project can access images
    from a specified directory. It then triggers a sync to load files.

    Args:
        base_url: Label Studio server URL.
        headers: HTTP headers with the API token.
        project_id: ID of the Label Studio project.
        project_name: Name of the project (used as the storage title).
        output_dir: Path to the local directory with image files.

    Returns:
        int: The storage ID if successful, otherwise None.
    """
    try:
        # Check if storage already exists
        storage_response = requests.get(
            f"{base_url}/api/storages/localfiles?project={project_id}",
            headers=headers
        )
        storage_response.raise_for_status()
        storages = storage_response.json()

        if storages:
            storage_id = storages[0]["id"]
            print(f"[✓] Using existing storage (ID: {storage_id})")
        else:
            document_root = str(Path("mock_io/data").resolve()).replace("\\", "/")
            print(f"[Debug] Setting up storage with path: {document_root}")

            # Create storage pointing to the parent directory
            storage_data = {
                "project": project_id,
                "title": f"{project_name} Local Storage",
                "path": document_root,  # Keep this at mock_io/data level
                "regex_filter": ".*\\.(jpg|jpeg|png)$",
                "use_blob_urls": False,
                "presign": False
            }

            storage_response = requests.post(
                f"{base_url}/api/storages/localfiles",
                headers=headers, 
                json=storage_data
            )

            # Print full response for debugging
            print(f"[Debug] Storage setup response: {storage_response.status_code}")
            if storage_response.status_code >= 400:
                print(f"[Debug] Error response: {storage_response.text}")
                print("[Warning] Could not set up local storage. Make sure Label Studio is running with:")
                print("[Warning] LOCAL_FILES_SERVING_ENABLED=true label-studio start")
                return None

            storage_response.raise_for_status()
            storage_id = storage_response.json()["id"]
            print(f"[✓] Created new storage (ID: {storage_id})")

        # Sync storage
        sync_response = requests.post(
            f"{base_url}/api/storages/localfiles/{storage_id}/sync",
            headers=headers
        )
        sync_response.raise_for_status()
        print("[✓] Storage sync initiated")
        return storage_id

    except requests.RequestException as e:
        print(f"[Error] Failed to set up storage: {e}")
        return None


def setup_label_studio(project_name: str, output_dir: str) -> dict:
    """Setup a Label Studio project for human review.

    Args:
        project_name: Name of the Label Studio project.
        output_dir: Directory to store Label Studio data.

    Returns:
        Dictionary with project configuration details.
    """
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not api_key:
        print("[Error] LABEL_STUDIO_API_KEY environment variable not set")
        return {}

    base_url = "http://localhost:8080"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "application/json"
    }

    print(f"Connecting to Label Studio at: {base_url}")

    # Find or create project
    project_id = _find_or_create_project(base_url, headers, project_name)
    if not project_id:
        return {}

    # Configure labeling interface
    _configure_interface(base_url, headers, project_id)

    # Setup storage
    storage_id = _connect_local_storage(base_url, headers, project_id, project_name, output_dir)

    return {
        "project_id": project_id,
        "storage_id": storage_id,
        "project_url": f"{base_url}/projects/{project_id}/data"
    }


def export_versioned_results(project_id: str, output_dir: Path, version: str = None) -> dict:
    """Export human review results to a versioned file.

    Args:
        project_id: Label Studio project identifier.
        output_dir: Directory to save human review results.
        version: Version string for the output file.

    Returns:
        Human review results.
    """
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not api_key:
        print("[Error] LABEL_STUDIO_API_KEY environment variable not set")
        return {}

    base_url = "http://localhost:8080"
    headers = {
        "Authorization": f"Token {api_key}"
    }

    try:
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get the list of tasks that have been labeled
        tasks_response = requests.get(
            f"{base_url}/api/projects/{project_id}/tasks",
            headers=headers
        )
        tasks_response.raise_for_status()
        tasks = tasks_response.json()

        # Tasks that have been labeled (for status updating)
        labeled_task_ids = {
            task["id"] for task in tasks 
            if task.get("annotations") and len(task["annotations"]) > 0
        }

        # Update status for labeled files
        for task in tasks:
            if task["id"] in labeled_task_ids:
                original_filename = task.get("data", {}).get("original_filename")
                if original_filename:
                    for json_file in mismatch_pending_dir.glob(f"*{original_filename}*"):
                        if _update_label_status(json_file, 2):
                            break

        # Export all results
        response = requests.get(
            f"{base_url}/api/projects/{project_id}/export?exportType=JSON",
            headers=headers
        )
        response.raise_for_status()

        results = response.json()

        # Remove base64 image data
        for result in results:
            if "data" in result and "image" in result["data"]:
                result["data"].pop("image", None)

        # Save results with version in filename
        results_path = output_dir / f"review_results_{version}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

        # Count how many of the results are labeled
        labeled_results_count = sum(1 for r in results if r.get("id") in labeled_task_ids)

        print("[✓] Exported results:")
        print(f"    - {len(results)} tasks ({labeled_results_count} labeled): {results_path}")

        # Transform results to labeled directory
        try:
            transformed_count = transform_reviewed_results_to_labeled(results, labeled_dir, image_dir)
            if transformed_count > 0:
                print(f"[✓] Auto-transformed {transformed_count} files to labeled directory")

        except Exception as e:
            print(f"[Warning] Failed to auto-transform results: {e}")

        return results

    except requests.exceptions.RequestException as e:
        print(f"[Error] Failed to export results: {e}")
        return {}


def _extract_confidence_from_results(export_result, bbox_index):
    """
    Extract confidence info from original prediction metadata.

    Args:
        export_result (dict): Single result from Label Studio export
        bbox_index (int): Index of the bounding box

    Returns:
        tuple: (confidence, confidence_flag)
    """
    annotations = export_result.get("annotations", [])
    if not annotations:
        return 1.0, "human"

    latest_annotation = annotations[-1]

    # Check if there's prediction data in the annotation
    if "prediction" in latest_annotation:
        prediction = latest_annotation["prediction"]
        result_items = prediction.get("result", [])
        if bbox_index < len(result_items):
            prediction_item = result_items[bbox_index]
            meta = prediction_item.get("meta", {})
            confidence = meta.get("confidence", 1.0)
            confidence_flag = meta.get("confidence_flag", "human")
            return confidence, confidence_flag

    return 1.0, "human"


def _transform_ls_result_to_original_format(export_result, image_dir: Path):
    """
    Transform Label Studio export result back to original JSON format.

    Args:
        export_result (dict): Single result from Label Studio export
        image_dir (Path): Directory containing images

    Returns:
        dict: Transformed data or None if failed
    """
    try:
        # Get annotations from export result
        annotations = export_result.get("annotations", [])
        if not annotations:
            return None

        latest_annotation = annotations[-1]
        if latest_annotation.get("was_cancelled", False):
            return None

        # Extract filename info
        original_filename = export_result["data"].get("original_filename")
        if not original_filename:
            return None

        # Transform annotations to original format
        predictions = []
        result_items = latest_annotation.get("result", [])

        for idx, annotation_item in enumerate(result_items):
            if annotation_item.get("type") == "rectanglelabels" and "value" in annotation_item:
                value = annotation_item["value"]
                labels = value.get("rectanglelabels", [])

                if not labels:
                    continue

                # Get image dimensions from annotation
                img_width = annotation_item.get("original_width")
                img_height = annotation_item.get("original_height")

                if not img_width or not img_height:
                    continue

                # Convert bbox from percentage to pixels
                bbox_pixels = _convert_bbox_from_percent(value, img_width, img_height)

                # Get confidence from original prediction
                confidence, confidence_flag = _extract_confidence_from_results(export_result, idx)

                prediction = {
                    "bbox": bbox_pixels,
                    "confidence": confidence,
                    "class": labels[0],
                    "confidence_flag": confidence_flag
                }
                predictions.append(prediction)

        return {
            "data": {
                "predictions": predictions,
                "label_status": 2,
                "reviewed_timestamp": latest_annotation.get("updated_at"),
                "annotation_id": latest_annotation.get("id"),
                "result_id": export_result.get("id")
            },
            "original_filename": original_filename
        }

    except Exception as e:
        print(f"[Error] Failed to transform {export_result.get('id')}: {e}")
        return None


def import_tasks_to_project(base_url: str, headers: Dict[str, str],
                            project_id: int, tasks_file: Path) -> bool:
    """
    Import tasks from a JSON file into a Label Studio project.

    Loads pre-annotated tasks into the given project for human review.

    Args:
        base_url: Label Studio server URL.
        headers: HTTP headers with the API token.
        project_id: ID of the target project.
        tasks_file: Path to the JSON file with tasks.

    Returns:
        bool: True if import succeeded, False otherwise.
    """

    try:
        with open(tasks_file, 'rb') as f:
            files = {'file': (tasks_file.name, f, 'application/json')}

            response = requests.post(
                f"{base_url}/api/projects/{project_id}/import",
                headers={k: v for k, v in headers.items() if k != 'Content-Type'},
                files=files
            )
            response.raise_for_status()
            print("[✓] Imported tasks to project")
            return True

    except (requests.RequestException, IOError) as e:
        print(f"[Error] Failed to import tasks: {e}")
        return False


def transform_reviewed_results_to_labeled(export_results: list,
                                          labeled_dir: Path = None,
                                          image_dir: Path = None) -> int:
    """
    Transform reviewed Label Studio results to labeled directory.
    Removes original pending files and saves human-reviewed data.

    Args:
        export_results (list): List of results from Label Studio export
        labeled_dir (Path): Directory to save labeled files
        image_dir (Path): Directory containing images

    Returns:
        int: Number of files transformed
    """
    if labeled_dir is None:
        labeled_dir = Path("mock_io/data/labeled")
    if image_dir is None:
        image_dir = Path("mock_io/data/raw/images")
    transformed_count = 0

    # Transform results to original format
    for export_result in export_results:
        if not export_result.get("annotations"):
            continue

        transformed = _transform_ls_result_to_original_format(export_result, image_dir)
        if not transformed:
            continue

        original_filename = transformed["original_filename"]
        labeled_file = labeled_dir / original_filename

        # Skip if file exists in labeled/
        if labeled_file.exists():
            continue

        try:
            # Save human-corrected data to labeled directory
            with open(labeled_file, 'w') as f:
                json.dump(transformed["data"], f, indent=2)

            # Remove original file from pending directory
            pending_file = mismatch_pending_dir / original_filename
            if pending_file.exists():
                pending_file.unlink()
            transformed_count += 1

        except Exception as e:
            print(f"[Error] Failed to save {original_filename}: {e}")

    print(f"[✓] Transformed {transformed_count} new labeled files")
    return transformed_count


def run_human_review(project_name: str = "AutoML-Human-Intervention",
                     export_results_flag: bool = None) -> dict:
    """Run the complete human review workflow.

    This function processes JSON files from the pending directory,
    creates a Label Studio project, and handles the human review workflow.

    Args:
        project_name: Name of the Label Studio project
        export_results_flag: Whether to export results (None = ask user)

    Returns:
        dict: Dictionary containing the results of the human review process.
    """
    # Initialize JSON files in the pending folder
    _initialize_json_files(mismatch_pending_dir)

    # Make sure Label Studio is running
    if not _ensure_label_studio_running():
        print("[Error] Could not start Label Studio")
        return {}

    # 1. Setup Label Studio project regardless of new files
    print("\nSetting up Label Studio project...")
    project_details = setup_label_studio(project_name, str(output_dir))

    if not project_details:
        print("[Error] Failed to set up Label Studio project")
        return {}

    # Check if there are files in the pending folder
    has_pending_files = any(mismatch_pending_dir.glob("*.json"))
    if not has_pending_files:
        print(f"[Warning] No JSON files found in {mismatch_pending_dir}")

    # 2. Generate tasks from pending files (with status=0)
    versioned_file, processed_files = _generate_ls_tasks(mismatch_pending_dir, image_dir, output_dir)

    # 3. Import new tasks if we have any
    if versioned_file and processed_files:
        api_key = os.getenv("LABEL_STUDIO_API_KEY")
        base_url = "http://localhost:8080"  
        headers = {
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json"
        }

        import_success = import_tasks_to_project(base_url, headers, project_details["project_id"], versioned_file)

        if import_success:
            _update_processed_files_status(processed_files)
        else:
            print("[Error] Task import failed")

    # 4. Notify user to complete review
    print("\n===== Label Studio Project Ready =====")
    print(f"Project URL: {project_details['project_url']}")

    should_export = export_results_flag
    if should_export is None:
        choice = input("\nDo you want to wait for review completion and export results now? (y/n): ")
        should_export = choice.lower() == 'y'

    if should_export:
        print("Please complete the human review in Label Studio, then press Enter to export results.")
        input()

        # 5. Export the human reviewed results with versioned filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = export_versioned_results(project_details["project_id"], reviewed_dir, timestamp)
        return results
    else:
        print(f"Project URL: {project_details['project_url']}")
        return project_details


if __name__ == "__main__":
    # Load environment variables including API key
    load_dotenv()

    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not api_key:
        print("Please set LABEL_STUDIO_API_KEY in the .env file")
        exit(1)

    export_flag = None
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        export_flag = True

    # Run the complete workflow
    results = run_human_review(export_results_flag=export_flag)

    if results:
        if isinstance(results, list):
            print("\n===== Human review completed and results exported! =====")
            print(f"Results saved to: {reviewed_dir}")
        else:
            # Show project_details
            print("\n===== Human review setup completed! =====")
            print(f"Project URL: {results.get('project_url')}")
    else:
        print("\n[Error] Human review process failed or was cancelled")
