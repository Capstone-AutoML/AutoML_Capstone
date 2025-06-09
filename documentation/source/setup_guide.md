# Setup Guide

This guide will walk you through setting up and running the AutoML CI/CD/CT: Continuous Training and Deployment Pipeline project.

## 1. Clone the Repository

```bash
git clone https://github.com/Capstone-AutoML/AutoML_Capstone.git
cd AutoML_Capstone
```

## 2. Run the Pipeline

### 💻 If You Have a GPU (CUDA Supported)

You can simply run:

```bash
docker compose up
```

This command will:

* Download necessary datasets and models on first run (unless `mock_io/data/`, `mock_io/data/distillation/`, or `mock_io/model_registry/model/` are removed).
* Automatically use your GPU **if** the following key is updated in **both** `train_config.json` and `pipeline_config.json`:

```json
"torch_device": "cuda"
```

> Default is `"cpu"`, which will force CPU-only execution.

---

### 💻 If You Have a CPU-Only Machine (No NVIDIA GPU)

Before running, **replace** your `docker-compose.yaml` file with:

```yaml
services:
  capstone:
    image: celt313/automl_capstone:v0.0.2
    container_name: automl_capstone
    shm_size: "4gb"
    working_dir: /app
    entrypoint: bash
    command: -c "source activate capstone_env && ./fetch_dataset.sh && python src/main.py"
    volumes:
      - .:/app

  generate_box:
    image: celt313/automl_capstone:v0.0.2
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && python src/generate_boxed_images.py"
    volumes:
      - .:/app

  human_intervention:
    image: celt313/automl_capstone:v0.0.2
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && python src/pipeline/human_intervention.py"
    volumes:
      - .:/app

  test:
    image: celt313/automl_capstone:v0.0.2
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && pytest tests/"
    volumes:
      - .:/app
```

Then run:

```bash
docker compose up
```

---

## 3. Run Tests (Optional)

To verify the setup and run unit tests:

```bash
docker compose run test
```

---

## 4. Generate Bounding Box Visualizations (Optional)

To run the script that overlays bounding boxes on sample and labeled images using predictions from YOLO, DINO, and mismatched sources:

```bash
docker compose run generate_box
```

---
This will:

* Sample and draw 10 images each from YOLO, DINO, and mismatched directories.

* Draw bounding boxes on all images from the labeled directory.

* Save the visualized outputs under `mock_io/boxed_images`

## 5. Human Review with Label Studio

For human-in-the-loop validation using Label Studio, refer to the [Human Intervention](human_in_loop.md) documentation section.

---

## 6. Configuration Files

These two config files control pipeline behavior:

* `train_config.json`: Training parameters, dataset paths, and device.
* `pipeline_config.json`: Pre-labeling, matching, augmentation, and distillation settings.

Defaults are generally sufficient, but GPU usage requires you to set:

```json
"torch_device": "cuda"
```

---

You're now ready to use the AutoML pipeline!
