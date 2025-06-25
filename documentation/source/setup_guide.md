# Setup Guide

This guide will walk you through setting up and running the AutoML CI/CD/CT: Continuous Training and Deployment Pipeline project.

## 1. Clone the Repository

```bash
git clone https://github.com/Capstone-AutoML/AutoML_Capstone.git
cd AutoML_Capstone
```

## 2. Run the Pipeline with Docker

**Important**: Docker cannot handle interactive Label Studio sessions for human review. Before running with Docker, you **must** disable human review in `automl_workspace/config/pipeline_config.json`:

```json
"process_options": {
  "skip_human_review": true
}
```

If you want to run human-in-the-loop validation using Label Studio, refer to the [Human Intervention](human_in_loop.md) documentation section.

### 💻 If You Have a GPU (CUDA Supported)

You can simply run:

```bash
docker compose up
```

This command will:

- Download necessary datasets and models on first run (unless `automl_workspace/data_pipeline/`, `automl_workspace/data_pipeline/distillation/`, or `automl_workspace/model_registry/model/` are removed).
- Automatically use your GPU **if** the following key is updated in **both** `automl_workspace/config/train_config.json` and `automl_workspace/config/pipeline_config.json`:

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
    image: celt313/automl_capstone:v0.0.3
    platform: linux/x86_64
    container_name: automl_capstone
    ipc: host
    working_dir: /app
    entrypoint: bash
    command: -c "source activate capstone_env && ./fetch_dataset.sh && python src/main.py"
    volumes:
      - .:/app

  generate_box:
    image: celt313/automl_capstone:v0.0.3
    platform: linux/x86_64
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && python src/generate_boxed_images.py"
    volumes:
      - .:/app

  human_intervention:
    image: celt313/automl_capstone:v0.0.3
    platform: linux/x86_64
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && python src/pipeline/human_intervention.py"
    volumes:
      - .:/app

  test:
    image: celt313/automl_capstone:v0.0.3
    platform: linux/x86_64
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

- Sample and draw 10 images each from YOLO, DINO, and mismatched directories.

- Draw bounding boxes on all images from the labeled directory.

- Save the visualized outputs under `automl_workspace/data_pipeline/boxed_images/`

---

## 5. Configuration Files

These two onfig files control pipeline behavior:

- `pipeline_config.json`: Process options, augmentation, and distillation settings.
- `train_config.json`: Training parameters, dataset paths, and device.
- `distillation_config.yaml`: Distillation settings (model paths, epochs, patience, etc.)
- `quantize_config.json` : Model quantization settings (labeled images paths, quantization method, etc.)

### Process Options

Control which pipeline steps to run via `pipeline_config.json`:

```json
// Set to true to skip a step
"process_options": {
  "skip_human_review": false,
  "skip_training": false,
  "skip_distillation": false,
  "skip_quantization": false
}
```

### Device Configuration

For GPU usage, set in **both** `pipeline_config.json` and `train_config.json`:

```json
"torch_device": "cuda"
```

Default is `"cpu"` for CPU-only execution.

---

## 6. Add Your Own Dataset

To start fresh with your own dataset:

1. **Clear existing data**:

   ```bash
   rm -rf automl_workspace/data_pipeline/*
   ```

2. **Add your images** to:

   ```text
   automl_workspace/data_pipeline/input/
   ├── image1.jpg
   ├── image2.jpg
   └── ...
   ```

Make sure the images are in `.jpg`, `.jpeg`, or `.png` format.
