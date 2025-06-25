# Data Fetching

This page outlines the unit tests for the `validate_input_images()` function in the `fetch_data` module, which ensures that the input directory contains valid image files before the pipeline proceeds.

---

## Overview

The `validate_input_images` function performs the following validations:

- Confirms the existence of the input directory
- Checks for at least one valid image file (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`)
- Raises a `ValueError` if the input directory is empty, nonexistent, or contains no valid images

---

## Fixtures

### `temp_dirs(tmp_path)`
Creates a temporary directory and populates it with a few dummy `.jpg` and `.png` files for positive test cases. Automatically cleaned up after test execution.

---

## Test Cases

### `test_validate_input_images_success`
**Purpose:**  
Confirms that the function passes silently when valid image files are present.

**Input:**  
A directory with `.jpg` and `.png` files.

**Expected Result:**  
No exception is raised.

---

### `test_validate_input_images_empty_directory`
**Purpose:**  
Checks behavior when the input directory is empty.

**Input:**  
An empty directory.

**Expected Result:**  
Raises `ValueError` with message:  
`"No images found in input directory"`

---

### `test_validate_input_images_no_valid_images`
**Purpose:**  
Validates that directories containing only non-image files are flagged.

**Input:**  
Files like `.txt`, `.csv`, and `.json`.

**Expected Result:**  
Raises `ValueError` with message:  
`"No images found in input directory"`

---

### `test_validate_input_images_mixed_files`
**Purpose:**  
Tests behavior when both image and non-image files are present.

**Input:**  
A mix of image files and other files.

**Expected Result:**  
No exception is raised.

---

### `test_validate_input_images_different_extensions`
**Purpose:**  
Ensures support for multiple common image file extensions.

**Input:**  
Files with extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`.

**Expected Result:**  
No exception is raised.

---

### `test_validate_input_images_nonexistent_directory`
**Purpose:**  
Handles the case when the input directory does not exist.

**Input:**  
A non-existent directory path.

**Expected Result:**  
Raises `ValueError` with message:  
`"No images found in input directory"`

---

## Summary

These tests ensure the robustness of the input image validation logic by covering a variety of directory states:
- Valid input
- Empty folders
- Nonexistent folders
- Mixed or incorrect file types
- Broad image extension support

All cases are automatically managed using `pytest` and `tmp_path` for isolation and reproducibility.
