# File Cleaning and Archiving

This page outlines the unit test coverage for the `clean_pipeline.py` module, which handles archiving labeled data and cleaning up the data pipeline workspace.

---

## Coverage Overview

The tests confirm that the cleaning process:

- Archives labeled JSON and associated image files into timestamped folders under the master dataset directory
- Cleans all pipeline folders except for `label_studio`
- Preserves the folder structure by keeping directories but removing their contents

---

## Fixtures

### `setup_clean_test_dirs`  
Creates a temporary mock workspace with:

- A `data_pipeline/` structure containing:
  - `labeled/`: with a sample JSON label
  - `input/`: with a matching image
  - `label_studio/`: preserved during cleanup
- A `master_dataset/` directory to hold archived results

---

## Function Tests

### `test_clean_pipeline_creates_archive_and_cleans`

- Verifies the creation of a timestamped archive folder containing:
  - `labels/`: where the original `.json` file is stored
  - `images/`: where the associated image file is copied
- Confirms that:
  - `label_studio/` folder and its contents remain untouched
  - All other folders (`labeled/`, `input/`, etc.) are retained but emptied

---

## Summary

This test ensures that the pipeline cleanup process effectively resets the workspace while preserving key folder structures and backing up labeled content to the `master_dataset/` archive.
