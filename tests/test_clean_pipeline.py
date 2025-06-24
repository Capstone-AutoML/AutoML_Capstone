"""
Test suite for `clean_pipeline.py`.

Covers:
- Archiving of JSON labels and matching images
- Cleaning all folders except 'label_studio'
- Preserving empty folder structure after cleaning
"""

import pytest
from src.pipeline.clean_pipeline import clean_pipeline_workspace


@pytest.fixture
def setup_clean_test_dirs(tmp_path):
    """
    Creates a mock workspace with labeled JSON and matching image in input.
    """
    data_pipeline = tmp_path / "data_pipeline"
    master_dataset = tmp_path / "master_dataset"
    label_studio = data_pipeline / "label_studio"
    labeled = data_pipeline / "labeled"
    input_dir = data_pipeline / "input"

    # Create directories
    for d in [label_studio, labeled, input_dir, master_dataset]:
        d.mkdir(parents=True, exist_ok=True)

    # Create matching label and image
    label_name = "sample_image.json"
    image_name = "sample_image.png"

    (labeled / label_name).write_text("{}")
    (input_dir / image_name).write_text("img")

    return data_pipeline, master_dataset, label_name, image_name


def test_clean_pipeline_creates_archive_and_cleans(setup_clean_test_dirs):
    data_pipeline_dir, master_dataset_dir, label_name, image_name = setup_clean_test_dirs

    clean_pipeline_workspace(data_pipeline_dir, master_dataset_dir)

    # Ensure archive folder is created
    archives = list(master_dataset_dir.glob("labeled_*"))
    assert len(archives) == 1

    labels_folder = archives[0] / "labels"
    images_folder = archives[0] / "images"

    # Ensure label and image are copied
    assert (labels_folder / label_name).exists()
    assert any(images_folder.glob("sample_image.*"))

    # Ensure only label_studio remains with contents, others are empty
    for folder in data_pipeline_dir.iterdir():
        if folder.name == "label_studio":
            assert folder.exists()
        else:
            # Folder exists but is empty
            assert folder.is_dir()
            assert not any(folder.iterdir())
