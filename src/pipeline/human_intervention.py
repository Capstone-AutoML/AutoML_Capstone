"""
Script for human review of the unmatched YOLO and SAM results.
"""
import json
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv


mismatch_dir = Path("mock_io/data/mismatched")
image_dir = Path("mock_io/data/raw/images")
output_dir = Path("mock_io/data/ls_tasks")
output_dir.mkdir(exist_ok=True)


def find_image_path(stem: str, image_dir: Path) -> Path:
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


def convert_bbox_to_percent(bbox, img_width, img_height):
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


def generate_ls_tasks(json_dir: Path, image_dir: Path, output_dir: Path):
    """
    Converts mismatched YOLOv8 prediction JSON files into a single
    Label Studio import file for human review.

    Args:
        json_dir (Path): Directory containing YOLO prediction `.json` files.
        image_dir (Path): Directory containing the corresponding image files.
        output_dir (Path): Path to save the generated `tasks.json` for LS.
    """
    tasks = []

    for json_file in json_dir.glob("*.json"):
        stem = json_file.stem
        image_path = find_image_path(stem, image_dir)

        if not image_path:
            print(f"[Skip] Image not found for: {stem}")
            continue

        try:
            with open(json_file, "r") as f:
                original = json.load(f)
        except Exception as e:
            print(f"[Error] Failed to read {json_file.name}: {e}")
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        prediction_data = []
        for item in original.get("predictions", []):
            box = convert_bbox_to_percent(item["bbox"], width, height)
            box.update({
                "rectanglelabels": [item["class"]]
            })
            prediction_data.append({
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": box
            })

        ls_mount_root = Path("mock_io/data").resolve()
        try:
            relative_path = image_path.resolve().relative_to(ls_mount_root)
        except ValueError:
            print(f"[Error] {image_path} is not under {ls_mount_root}")
            continue

        task = {
            "data": {
                "image": f"/data/local-files/?d={relative_path}"
            }
        }

        if prediction_data:
            task["predictions"] = [{
                "model_version": "yolov8-prelabel-mismatch",
                "result": prediction_data
            }]

        tasks.append(task)

    with open(output_dir, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"[√] Exported {len(tasks)} tasks to {output_dir}")


def setup_label_studio(project_name: str, output_dir: str) -> dict:
    """Setup a Label Studio project for human review.

    Args:
        project_name: Name of the Label Studio project.
        output_dir: Directory to store Label Studio data.

    Returns:
        Dictionary with project configuration details.
    """
    pass


def prepare_review_data(images_dir: str, yolo_results: dict, sam_results: dict) -> str:
    """Prepare data where YOLO and SAM disagree for human review.

    Args:
        images_dir: Directory containing original images.
        yolo_results: YOLO detection results by image filename.
        sam_results: SAM segmentation results by image filename.

    Returns:
        Path to prepared review data directory.
    """
    pass


def create_review_interface(project_id: str) -> bool:
    """Configure the side-by-side image review interface in Label Studio.

    Args:
        project_id: Label Studio project identifier.

    Returns:
        True if successful, False otherwise.
    """
    pass


def export_results(project_id: str, output_dir: str) -> dict:
    """Export human review decision results.

    Args:
        project_id: Label Studio project identifier.
        output_dir: Directory to save human review results.

    Returns:
        Human review results.
    """
    pass


def run_human_review(images_dir: str, yolo_results: dict, sam_results: dict, output_dir: str) -> dict:
    """Run the complete human review workflow.

    Args:
        images_dir: Directory containing the original images.
        yolo_results: YOLO detection results.
        sam_results: SAM segmentation results.
        output_dir: Directory for all outputs.

    Returns:
        dict: Dictionary containing the results of the human review process.
    """
    pass


if __name__ == "__main__":
    # Example usage of the functions defined above
    # The label studio API key should be set in .env file
    load_dotenv()
    pass
