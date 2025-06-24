# File Cleaning and Archiving

This module provides a utility for archiving labeled data and cleaning the workspace after each labeling cycle. It ensures that the project directory remains organized while preserving important folders such as `label_studio`.

---

## Function: `clean_pipeline_workspace`

```python
clean_pipeline_workspace(data_pipeline_dir: Path, master_dataset_dir: Path)
```

### Description

Archives labeled results and cleans the data pipeline workspace.

- Copies `.json` label files from `data_pipeline/labeled/` to a timestamped archive under `master_dataset/`.
- Copies matching image files from `data_pipeline/input/` into the same archive (based on file stem).
- Clears all folders in `data_pipeline/` except `label_studio`.

### Arguments

| Name                | Type | Description                                                  |
|---------------------|------|--------------------------------------------------------------|
| `data_pipeline_dir` | Path | Path to the root of the `data_pipeline/` directory           |
| `master_dataset_dir`| Path | Path to the `master_dataset/` directory for saving archives  |

---

## Folder Structure After Cleaning

Only the following directory is preserved:

```
data_pipeline/
├── input/           # Emptied
├── labeled/         # Emptied
├── label_studio/    # Preserved with all contents
└── ...              # All other folders are emptied
```

---

## Archive Output Example

A successful run creates an archive folder like this:

```
master_dataset/
└── labeled_20250624_140501/
    ├── labels/
    │   ├── image1.json
    │   ├── image2.json
    └── images/
        ├── image1.png
        ├── image2.jpg
```

---

## Notes

- Matching images are copied based on the label file’s stem (e.g., `image1.json` matches `image1.png`, `image1.jpg`, etc.).
- If a matching image is not found, a warning is printed.
- Preserves folder structure for any future uploads to Label Studio or audit tracking.
