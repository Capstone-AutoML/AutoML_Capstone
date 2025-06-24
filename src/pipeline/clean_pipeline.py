from pathlib import Path
from datetime import datetime
import shutil

def clean_pipeline_workspace(data_pipeline_dir: Path, master_dataset_dir: Path):
    """
    Archives labeled results and cleans the workspace except for label_studio.

    - Copies JSON label files from data_pipeline/labeled to a timestamped folder in master_dataset.
    - Copies matching images from data_pipeline/input (regardless of extension).
    - Cleans all folders in data_pipeline except 'label_studio'.
    """

    labeled_path = data_pipeline_dir / "labeled"
    input_path = data_pipeline_dir / "input"
    preserve_folders = {"label_studio"}

    # Step 1: Archive labeled results
    if labeled_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = master_dataset_dir / f"labeled_{timestamp}"
        labels_dir = archive_dir / "labels"
        images_dir = archive_dir / "images"
        labels_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        for json_file in labeled_path.glob("*.json"):
            # Copy label file
            shutil.copy(json_file, labels_dir / json_file.name)

            # Match image file by stem (remove `.json`) and look for any extension in input/
            image_stem = json_file.stem  # keep full stem
            matching_images = list(input_path.glob(f"{image_stem}.*"))

            if matching_images:
                shutil.copy(matching_images[0], images_dir / matching_images[0].name)
            else:
                print(f"[⚠] Image not found for label {json_file.name}: expected something like {image_stem}.*")

        print(f"[✓] Archived labeled data to {archive_dir}")
        print(f"[✓] Total archived: {len(list(labels_dir.glob('*.json')))} labels")

    else:
        print("[Info] No labeled folder found to archive.")

    # Step 2: Clean contents of all folders in data_pipeline except preserved ones
    for folder in data_pipeline_dir.iterdir():
        if folder.is_dir() and folder.name not in preserve_folders:
            for item in folder.iterdir():
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)

