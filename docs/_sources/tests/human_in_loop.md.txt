# Human-in-the-Loop

This page outlines the unit test coverage for the `human_intervention.py` script, which manages the human-in-the-loop review process using Label Studio. It includes utilities for task preparation, status tracking, project setup, and final result transformation.

---

## Coverage Overview

This suite tests key internal functions as well as the high-level `run_human_review()` workflow. It verifies behavior related to:

- Image and JSON handling
- Bounding box transformation
- Task generation and import
- Label Studio integration
- Export and transformation of reviewed results

---

## Fixtures

### `temp_dirs`  
Creates test directories for:
- Raw images (`images/`)
- Prediction JSONs (`json/`)
- Task outputs (`output/`)
- Final labeled data (`labeled/`)

Also generates test image and JSON files.

### `mock_global_directories`  
Patches global directory variables in the module to avoid side effects during testing.

---

## Key Function Tests

### Image and File Utilities

- `test_find_image_path`: Validates lookup of `.jpg` or `.png` image by name.
- `test_update_label_status`, `test_update_label_status_file_not_found`: Updates the `label_status` in prediction JSONs and handles missing files.
- `test_initialize_json_files`: Adds `label_status = 0` to uninitialized prediction files.
- `test_initialize_json_files_invalid_json`: Handles corrupt or non-parsable JSON gracefully.

### Bounding Box Conversion

- `test_convert_bbox_to_percent`: Converts pixel coordinates to percentages for Label Studio.
- `test_convert_bbox_from_percent`: Converts normalized coordinates back to absolute pixels.

### Task Generation

- `test_generate_ls_tasks`: Generates Label Studio tasks from predictions and images, including prediction overlays.
- `test_generate_ls_tasks_empty_dir`, `test_generate_ls_tasks_missing_images`: Validates behavior when source files are missing or empty.

### Processed File Status

- `test_update_processed_files_status`: Marks all JSONs as imported by setting `label_status = 1`.

### Label Studio Integration

- `test_ensure_label_studio_running`, `test_ensure_label_studio_running_needs_start`: Verifies Label Studio is running or starts it if needed.
- `test_find_or_create_project`: Finds or creates a project via API.
- `test_configure_interface`: Updates the labeling interface for the selected project.
- `test_import_tasks_to_project`: Posts generated tasks to Label Studio.

### Project Setup

- `test_setup_label_studio`: Sets up the project and returns configuration for further API use.

### Exporting and Transforming Results

- `test_export_versioned_results`: Downloads reviewed results and removes base64-encoded image data.
- `test_transform_ls_result_to_original_format`: Converts reviewed annotations back to the format expected by the labeling pipeline.
- `test_extract_confidence_from_results`: Extracts model confidence and reviewer flags from results.
- `test_transform_reviewed_results_to_labeled`: Moves approved predictions to the `labeled/` directory and deletes pending files.

### Full Workflow: `run_human_review()`

- `test_run_human_review`: Executes the entire human review loop (task generation, upload, status updates).
- `test_run_human_review_with_export`: Includes export flow and transformation of reviewed annotations.
- `test_run_human_review_setup_failure`: Handles failed setup conditions.
- `test_run_human_review_no_label_studio`: Aborts workflow if Label Studio is not running.

---

## Summary

This page outlines a comprehensive test suite covering:

- File preparation and cleanup  
- Label Studio setup and API interactions  
- Status tracking and prediction transformation  
- Edge cases such as empty inputs, bad JSONs, or API failures

Together, these tests ensure that the human-in-the-loop review flow is robust, traceable, and safe to integrate into automated pipelines.
