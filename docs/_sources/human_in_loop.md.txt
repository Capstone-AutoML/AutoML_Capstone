# Human-in-the-Loop

This module handles the **human intervention workflow** for reviewing mismatches between YOLO and Grounding DINO predictions. It connects to **Label Studio**, converts pre-labeling data into importable tasks, and tracks the status of each review round.

---

## Key Capabilities

- Detects and processes prediction mismatches
- Converts bounding boxes into Label Studio format
- Launches and configures Label Studio locally
- Tracks human review progress using `label_status`
- Exports labeled results in versioned JSON files

---

## Setup Instructions

### 1. Environment Setup

**Step 1**: Ensure that required dependencies are installed

If you haven't installed all dependencies in the `environment.yml` or you only want to go through the human-in-the-loop process, install the dependencies in the `human_review_env.yml`:

```bash
conda env create -f human_review_env.yml
conda activate human_review_env
```

**Step 2**: Set up the Label Studio API key

1. Start Label Studio: `label-studio start`.
2. Create an account and login.
3. In the web UI, go to: **☰ Hamburger menu** → **Organization** → **API Token Settings**.
4. If **Legacy Tokens** are not enabled, turn them on.
5. Then navigate to: Top-right → **Account & Settings** → **Legacy Token**.
6. Copy the token and create a `.env` file in the project root with the following content: `LABEL_STUDIO_API_KEY=your_token_here`.

---

### 2. Pipeline Configuration

Configure human review in `automl_workspace/config/pipeline_config.json`:

```json
{
  "process_options": {
    "skip_human_review": false // Set to true to skip human review
  }
}
```

---

### 3. Directory Setup

The following folders are initialized automatically if not present:

```bash
AutoML_Capstone/
├── .env                   # Put your API key here
└── automl_workspace/data_pipeline/
    ├── input/             # All referenced images
    ├── label_studio/
    │   ├── pending/       # Mismatches (raw YOLO output JSONs)
    │   ├── tasks/         # Temporary task JSONs for import
    │   └── results/       # Human-reviewed output ends up here
    └── labeled/           # Final labeled data
```

## Workflow

### 1. Status Tracking with `label_status`

Each JSON file in the `pending/` folder includes a `label_status` field that tracks its progress through the review pipeline:

| `label_status` | Description                                |
| -------------- | ------------------------------------------ |
| `0`            | Unprocessed - ready to be imported         |
| `1`            | Imported to Label Studio, pending labeling |
| `2`            | Human-reviewed and labeled                 |

This status field is updated automatically by the script:

- New files (without `label_status`) are assigned `0`.
- Once imported into Label Studio, status becomes `1`.
- After review and export, it updates to `2`.

This makes it easy to resume or rerun reviews without duplicating work.

---

### 2. Main Functions

#### `_initialize_json_files()`

- Scans pending directory for new JSON files
- Sets `label_status = 0` for files without status field
- Prepares files for import to Label Studio

#### `_generate_ls_tasks()`

- Converts pending JSON predictions + images into Label Studio tasks
- Encodes images to base64 for web display
- Saves versioned task file: `tasks_YYYYMMDD_HHMMSS.json`

#### `_ensure_label_studio_running()`

- Checks if Label Studio is live on `localhost:8080`
- Starts it as a subprocess if not running

#### `setup_label_studio()`

- Connects to Label Studio via API
- Creates or reuses a project by name
- Configures the labeling interface with bounding box tools
- Connects to local image folder for task visualization

#### `import_tasks_to_project()`

- Uploads tasks to Label Studio for human labeling

#### `export_versioned_results()`

- Exports all reviewed tasks from Label Studio
- Saves versioned result file: `review_results_YYYYMMDD_HHMMSS.json`
- Updates `label_status` to `2` for completed tasks

#### `transform_reviewed_results_to_labeled()`

- Converts Label Studio export format back to original JSON structure
- Moves completed files from `pending/` to `labeled/` directory
- Enables automatic pipeline continuation to training step

---

## Example Usage

### 1. Place Files for Review

- Mismatched JSON files are automatically placed in `automl_workspace/data_pipeline/label_studio/pending/`
- Ensure referenced images exist in `automl_workspace/data_pipeline/input/`

### 2. Run the Review Process

**Option A**: Run the full AutoML pipeline with human review enabled

```json
// Configure in pipeline_config.json
"process_options": {
  "skip_human_review": false
}
```

```bash
# Run main pipeline
python src/main.py
```

**Option B**: Run human review independently

```bash
# Run without export
python src/pipeline/human_intervention.py

# Run and immediately export after human review
python src/pipeline/human_intervention.py --export
```

### 3. Review in Label Studio

- Follow the URL shown in terminal (`http://localhost:8080/projects/...`)
- Review bounding boxes and assign labels
- Press Enter in terminal to finish once labeling is done

---

## Input and Output

### Example JSON (YOLO prediction output)

```json
{
  "predictions": [
    {
      "bbox": [
        51.57196044921875, 165.7647247314453, 402.517578125, 459.77508544921875
      ],
      "confidence": 0.597667932510376,
      "class": "FireBSI"
    }
  ],
  "label_status": 0 // 0 = unimported, 1 = imported (unreviewed), 2 = human reviewed
}
```

- `bbox`: Format is `[x_min, y_min, x_max, y_max]` in pixels
- Final reviewed results will be saved under `automl_workspace/data_pipeline/label_studio/results/`
- Processed files automatically move to `automl_workspace/data_pipeline/labeled/` for training
