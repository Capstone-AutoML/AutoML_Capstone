# YOLO Prelabeling Tests

This document describes the test suite for the YOLO prelabeling module, which is responsible for automatically generating initial labels for unlabeled images in the wildfire detection pipeline. YOLO prelabeling serves as the first step in the semi-automated labeling process, providing AI-generated predictions that will later be refined through human review.

## Overview

The YOLO prelabeling tests verify the functionality of the automated labeling pipeline that processes raw images and generates JSON files containing bounding box predictions, confidence scores, and class labels. This module is critical for the human-in-the-loop labeling workflow, where AI-generated labels reduce the manual labeling burden while maintaining quality through human verification.

## Test Structure

The test file `tests/test_yolo_prelabelling.py` contains comprehensive tests covering all aspects of the prelabeling pipeline:

### Mock Components

#### `DummyYOLOModel`
- **Purpose**: Mock YOLO model that simulates real model behavior without loading actual weights
- **Features**: Returns consistent dummy predictions for testing
- **Predictions**: Two mock detections per image (fire and smoke classes)
- **Usage**: Enables testing without requiring actual trained models

### Fixtures

#### `patch_yolo_model`
- **Purpose**: Patches the YOLO import to return the dummy model
- **Usage**: Ensures all tests use the mock model for consistent, fast testing

#### `tmp_dirs_with_images`
- **Purpose**: Creates temporary directories with test images
- **Contents**: 
  - Valid test images (JPG and PNG formats)
  - Corrupted image file
  - Non-image text file
- **Usage**: Provides realistic test data for file processing tests

#### `mock_model_path`
- **Purpose**: Creates a mock model file path
- **Usage**: Simulates model file existence for loading tests

## Individual Tests

### File Processing Tests

#### `test_get_image_files_valid`
**Purpose**: Tests that image file discovery works correctly.

**What it tests**:
- Verifies that only files with valid image extensions are found
- Ensures corrupted images are included (since filtering is based on extension only)
- Confirms that non-image files are excluded

**Key aspects**:
- Tests JPG, PNG, and JPEG extensions
- Handles case-insensitive extension matching
- Returns correct number of image files (3 in test case)

**Importance**: This function is the first step in the pipeline, ensuring only appropriate files are processed.

#### `test_get_image_files_empty_directory`
**Purpose**: Tests edge case handling for empty directories.

**What it tests**:
- Verifies that empty directories return empty lists
- Ensures no errors occur when no files are present

**Importance**: Prevents pipeline failures when no images are available for processing.

### Model Loading Tests

#### `test_load_model_success`
**Purpose**: Tests successful model loading and device placement.

**What it tests**:
- Verifies that YOLO models load correctly from valid paths
- Ensures models are moved to the specified device
- Confirms proper arguments are passed to YOLO constructor

**Key aspects**:
- Tests model loading with mock model path
- Verifies device placement (CPU in test case)
- Ensures proper error handling for successful cases

**Importance**: Model loading is critical for the entire prelabeling process.

#### `test_load_model_file_not_found`
**Purpose**: Tests error handling for missing model files.

**What it tests**:
- Verifies that FileNotFoundError is raised for non-existent models
- Ensures clear error messages are provided
- Tests fast failure to prevent silent errors

**Importance**: Prevents pipeline failures when model files are missing or corrupted.

### Prediction Processing Tests

#### `test_process_prediction`
**Purpose**: Tests the conversion of YOLO predictions to standardized format.

**What it tests**:
- Verifies correct parsing of bounding box coordinates
- Tests confidence score extraction
- Ensures proper class name mapping
- Validates output format for multiple detections

**Key aspects**:
- Processes multiple detections per image
- Converts coordinates to list format
- Maps class IDs to class names
- Returns structured prediction dictionaries

**Output format tested**:
```json
{
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.8,
  "class": "fire"
}
```

**Importance**: This function standardizes YOLO output for downstream processing and human review.

#### `test_process_prediction_empty`
**Purpose**: Tests handling of images with no detections.

**What it tests**:
- Verifies that empty prediction lists are handled gracefully
- Ensures no errors occur when no objects are detected
- Returns empty list instead of failing

**Importance**: Many images may not contain wildfire-related objects, so this edge case is common.

### Output Generation Tests

#### `test_save_predictions`
**Purpose**: Tests JSON file generation and data persistence.

**What it tests**:
- Verifies that predictions are saved to JSON files
- Ensures proper file structure and formatting
- Tests data integrity and completeness

**Key aspects**:
- Creates output files in specified locations
- Validates JSON format and structure
- Ensures all prediction data is preserved

**Importance**: Output files are used by the human review interface and downstream pipeline components.

### Main Pipeline Tests

#### `test_generate_yolo_prelabelling_success`
**Purpose**: Tests the complete prelabeling pipeline with valid inputs.

**What it tests**:
- Verifies end-to-end processing of image files
- Ensures JSON outputs are created for all processed images
- Tests pipeline integration and data flow

**Key aspects**:
- Processes multiple image formats
- Creates corresponding JSON output files
- Validates output file structure and content
- Tests successful processing statistics

**Importance**: This is the main integration test ensuring the entire pipeline works correctly.

#### `test_generate_yolo_prelabelling_device_auto`
**Purpose**: Tests automatic device detection functionality.

**What it tests**:
- Verifies that 'auto' device setting triggers device detection
- Ensures models are loaded on the detected device
- Tests device auto-detection integration

**Importance**: Enables cross-platform compatibility and optimal device utilization.

#### `test_generate_yolo_prelabelling_handles_corrupted_images`
**Purpose**: Tests robustness when processing corrupted or unreadable images.

**What it tests**:
- Verifies that corrupted images don't crash the pipeline
- Ensures processing continues for other valid images
- Tests error handling and recovery

**Importance**: Real-world datasets often contain corrupted files that shouldn't break the pipeline.

#### `test_generate_yolo_prelabelling_model_error`
**Purpose**: Tests error handling for model loading failures.

**What it tests**:
- Verifies that missing model files cause appropriate errors
- Ensures pipeline fails fast with clear error messages
- Tests error propagation

**Importance**: Prevents silent failures and provides clear feedback for debugging.

#### `test_generate_yolo_prelabelling_empty_directory`
**Purpose**: Tests behavior when no images are available.

**What it tests**:
- Verifies that empty input directories don't cause errors
- Ensures no output files are created when no input exists
- Tests graceful handling of edge cases

**Importance**: Handles scenarios where datasets are empty or images are moved/deleted.

#### `test_generate_yolo_prelabelling_verbose_mode`
**Purpose**: Tests verbose logging and progress reporting.

**What it tests**:
- Verifies that verbose mode prints expected log messages
- Tests device information, model loading, and progress updates
- Ensures summary statistics are reported

**Key log messages tested**:
- Device information
- Model loading confirmation
- Image count and processing progress
- Success/failure statistics

**Importance**: Provides visibility into pipeline progress and debugging information.

#### `test_generate_yolo_prelabelling_creates_output_directory`
**Purpose**: Tests automatic output directory creation.

**What it tests**:
- Verifies that missing output directories are created automatically
- Ensures pipeline doesn't fail due to missing paths
- Tests directory creation with nested paths

**Importance**: Improves user experience by handling common setup issues automatically.

#### `test_generate_yolo_prelabelling_pytorch_mps_fallback`
**Purpose**: Tests Apple Silicon compatibility and MPS fallback.

**What it tests**:
- Verifies that PYTORCH_ENABLE_MPS_FALLBACK environment variable is set
- Ensures compatibility with Apple Silicon Macs
- Tests environment variable management

**Importance**: Enables the pipeline to work on Apple Silicon devices with proper PyTorch support.

## Test Coverage

The YOLO prelabeling tests provide comprehensive coverage of:

1. **File Management**: Image discovery, validation, and processing
2. **Model Operations**: Loading, device placement, and error handling
3. **Data Processing**: Prediction parsing and standardization
4. **Output Generation**: JSON file creation and data persistence
5. **Error Handling**: Robust error management and recovery
6. **Integration**: End-to-end pipeline functionality
7. **Platform Compatibility**: Cross-platform device detection and support

## Integration with Pipeline

These tests ensure that the YOLO prelabeling module integrates properly with the broader wildfire detection pipeline:

- **Automated Labeling**: Provides initial AI-generated labels for human review
- **Data Quality**: Ensures consistent output format for downstream components
- **Error Resilience**: Maintains pipeline stability with corrupted or missing data
- **Performance**: Optimizes device usage and processing efficiency
- **User Experience**: Provides clear feedback and progress reporting

## Running the Tests

To run the YOLO prelabeling tests:

```bash
# Run all YOLO prelabeling tests
pytest tests/test_yolo_prelabelling.py -v

# Run a specific test
pytest tests/test_yolo_prelabelling.py::test_generate_yolo_prelabelling_success -v

# Run with coverage
pytest tests/test_yolo_prelabelling.py --cov=pipeline.prelabelling
```

## Dependencies

The tests require:
- PyTorch for model operations
- Ultralytics YOLO for model handling
- OpenCV for image processing
- NumPy for numerical operations
- Pytest for test framework
- Mock for test isolation

## Notes

- Tests use mock models to avoid loading large pre-trained weights during testing
- Dummy images are created programmatically to avoid requiring actual wildfire images
- All tests are designed to run quickly and independently for efficient CI/CD integration
- Error handling tests ensure the pipeline is robust to real-world data issues
- Platform compatibility tests enable cross-platform deployment

