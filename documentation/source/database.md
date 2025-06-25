# Database Schema Overview

This page outlines the database schema designed to support the core components of the wildfire detection pipeline. The schema is structured for scalability and maintainability, with a focus on enabling post-handoff use by the partner team.

---

## `images` Table

Tracks all image assets in the pipeline.

| Field             | Type         | Description                                         |
|------------------|--------------|-----------------------------------------------------|
| `id`             | String (UUID) | Unique image identifier                            |
| `image_path`     | String        | Local path to the image file                        |
| `label_path`     | String        | Local path to the corresponding label JSON file     |
| `classes`        | JSON          | Array of class labels (e.g., `["Fire", "Smoke"]`)   |
| `bboxes`         | JSON          | Array of bounding boxes (`[[x1, y1, x2, y2], ...]`) |
| `label_status`   | Enum          | One of `auto`, `human`, or `mismatch`              |
| `semi_supervised`| JSON          | Additional info from pre-labeling (if any)         |
| `tags`           | JSON          | Array of tags (e.g., `["augmented", "smoke"]`)     |

---

## `models` Table

Stores metadata for each trained, distilled, or quantized model.

| Field             | Type         | Description                                     |
|------------------|--------------|-------------------------------------------------|
| `id`             | String (UUID) | Model identifier                               |
| `model_path`     | String        | Path to the model weights file                 |
| `model_type`     | Enum          | One of `base`, `distilled`, or `quantized`     |
| `model_status`   | Enum          | `training` or `trained`                        |
| `train_image_ids`| JSON          | UUIDs of images used for training              |
| `val_image_ids`  | JSON          | UUIDs of images used for validation            |
| `test_image_ids` | JSON          | UUIDs of images used for testing               |
| `training_history`| JSON         | Training metrics, loss, and other metadata     |

---

## `experiments` Table

Tracks experiment configurations and evaluation metrics.

| Field         | Type         | Description                                          |
|---------------|--------------|------------------------------------------------------|
| `id`         | String (UUID) | Unique experiment identifier                         |
| `model_id`   | String        | Foreign key to `models.id`                           |
| `augmentation` | JSON        | Augmentation settings (e.g., `{"flip": true}`)       |
| `random_seed` | String       | Seed used for reproducibility                        |
| `performance` | JSON         | Model performance metrics (e.g., `{"mAP": 0.89}`)    |
| `config`      | JSON         | Training configuration (e.g., `{"batch_size": 16}`)  |

---

## `.env` Configuration

Set up your database connection in a `.env` file at the root of the project:

```env
# Default SQLite setup
DATABASE_URL=sqlite:///db.sqlite3

# Optional PostgreSQL setup
# DATABASE_URL=postgresql://username:password@localhost:5432/your_database
