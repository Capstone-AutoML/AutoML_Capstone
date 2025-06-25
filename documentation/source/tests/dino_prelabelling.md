# Grounding DINO Prelabeling

This page outlines the unit test coverage for the `gdino_prelabelling.py` script. The module runs object detection on a directory of images using the Grounding DINO model and saves prediction results in JSON format.

---

## Coverage Overview

This test suite includes validation for:

- File scanning via `_get_image_files`
- End-to-end behavior of `generate_gd_prelabelling`
- Device detection and fallback logic
- Handling of corrupted or invalid image files
- Handling of empty or missing directories
- JSON output structure and formatting

---

## Constants and Configs

The following are tested either directly or via mock configuration injection:

- `TEXT_PROMPTS`
- `BOX_THRESHOLD`
- `TEXT_THRESHOLD`
- Device resolution: `"cuda"`, `"cpu"`, `"auto"`
- Paths to model weights and config files

---

## Functions Tested

### `_get_image_files(directory)`

- **Returns**: Valid image paths (`.jpg`, `.jpeg`, `.png`)
- **Tests**:
  - Valid image files
  - Mixed content (image + non-image)
  - Empty directories
  - Nonexistent directory (expected to raise error upstream)

---

### `generate_gd_prelabelling(...)`

**Core test focus:**

- Successful predictions written to JSON
- Skipped or unreadable files are logged
- Predictions include class, confidence, bounding box
- All output files use expected format
- Handles model loading errors
- Applies config thresholds correctly
- Automatically creates output directories if missing
- Uses fallback device detection logic when set to `"auto"`

---

## Key Edge Cases Tested

- **Empty folder**: Returns no predictions and exits gracefully
- **Non-image files**: Skipped without error
- **Corrupted images**: Skipped and logged
- **Missing model file**: Raises `FileNotFoundError`
- **Output folder does not exist**: Auto-created
- **Multiple valid predictions per image**: Verified in output structure
- **Verbose mode**: Confirms detailed logs are printed to stdout

---

## Example Test Assertions

- Count of processed images matches input count
- Prediction JSON contains required keys: `bbox`, `confidence`, `class`, `source`
- Output filenames correspond to input image names
- Invalid images do not result in written JSON
- `torch_device` settings are passed to model correctly

---

## Summary

This test suite ensures that `gdino_prelabelling.py`:

- Works as expected across different environments (CPU, GPU, MPS)
- Produces consistently formatted output
- Handles missing, corrupted, or unexpected files gracefully
- Is configurable via the pipeline-level `config` and can be extended

The use of mocks and temporary directories isolates test behavior from model internals, ensuring that unit-level functionality is verified in a reproducible way.
