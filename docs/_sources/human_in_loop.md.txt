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

## Workflow Outline

### 1. Directory Setup

The following folders are initialized automatically if not present:

- `mock_io/data/mismatched/pending`: YOLO predictions needing review
- `mock_io/data/mismatched/reviewed_results`: Final output directory
- `mock_io/data/ls_tasks`: Temporary task export folder

---

### 2. Label Status Tracking

The system uses `label_status` in each JSON file to track processing stages:

- `0`: Unprocessed (ready to import)
- `1`: Imported to Label Studio
- `2`: Labeled by a human

---

### 3. Main Functions

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

---

### 4. Complete Workflow

Run the full review pipeline via:

```python
run_human_review(project_name="AutoML-Human-Intervention")
```

Or export immediately after review:

```python
run_human_review(project_name="AutoML-Human-Intervention", export_results_flag=True)
```

---

## Environment Setup

Make sure to add your Label Studio API token to a `.env` file:

```
LABEL_STUDIO_API_KEY=your_api_key_here
```

Install Label Studio (if not already):

```bash
pip install label-studio
```

Start it with local file access enabled:

```bash
LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true label-studio start
```

---

## Output Summary

- Tasks successfully processed and imported
- Project accessible at `http://localhost:8080/projects/{project_id}/data`
- Final reviewed results saved under `mock_io/data/mismatched/reviewed_results`
