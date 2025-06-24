# Directory Setup

This page outlines the unit test coverage for the `directory_setup.py` module, which is responsible for initializing the `automl_workspace` directory structure.

---

## Coverage Overview

The tests validate that the function `create_automl_workspace()`:

- Creates a well-defined directory tree for data pipeline, model registry, config, and label studio
- Supports custom base paths
- Gracefully handles already existing directories
- Responds to permission errors with informative messages
- Properly constructs nested directories as intended

---

## Function Tests

### `test_create_automl_workspace_default_path`
Verifies that the default workspace is created correctly when no custom path is provided.

### `test_create_automl_workspace_custom_path`
Checks that the workspace is created under a user-specified base directory.

### `test_create_automl_workspace_already_exists`
Confirms the function does not overwrite existing files and directories when re-run.

### `test_create_automl_workspace_permission_error`
Simulates a `PermissionError` during directory creation and verifies that the error is logged appropriately.

### `test_create_automl_workspace_nested_structure`
Ensures that deeply nested directories like `label_studio/pending` are created as expected.

### `test_create_automl_workspace_makedirs_called_correctly`
Mocks `os.makedirs` and asserts it is called for all required subdirectories.

---

## Summary

These tests guarantee that `create_automl_workspace()` reliably sets up the workspace environment in various conditions, making the AutoML pipeline reproducible and structured across different setups.
