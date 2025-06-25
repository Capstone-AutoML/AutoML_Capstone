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

If you want to run the auto-labeling part of the pipeline separately, do:

```bash
docker compose run auto_labeling
```

> This step should always come first.

Then, to run the augmentation, training, and compression steps, use:

```bash
docker compose run train_compress
```

### 💻 If You Have a CPU-Only Machine (No NVIDIA GPU)

Before running, **replace** your `docker-compose.yaml` file with:

```yaml
services:
  capstone:
    image: celt313/automl_capstone:v0.0.3
    ipc: host
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
    ipc: host
    platform: linux/x86_64
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && python src/generate_boxed_images.py"
    volumes:
      - .:/app
  
  auto_labeling:
    image: celt313/automl_capstone:v0.0.3
    ipc: host
    platform: linux/x86_64
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && ./fetch_dataset.sh && python src/label_main.py"
    volumes:
      - .:/app


  train_compress:
    image: celt313/automl_capstone:v0.0.3
    ipc: host
    platform: linux/x86_64
    profiles: ["optional"]
    entrypoint: bash
    command: -c "source activate capstone_env && python src/train_compress.py"
    volumes:
      - .:/app

  test:
    image: celt313/automl_capstone:v0.0.3
    ipc: host
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

to run the entire pipeline.

If you want to run the auto-labeling part of the pipeline separately, do:

```bash
docker compose run auto_labeling
```

> This step should always come first.

Then, to run the augmentation, training, and compression steps, use:

```bash
docker compose run train_compress
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

These config files control pipeline behavior:

- `pipeline_config.json`: Process options and distillation settings.
- `augmentation_config.json`: Augmentation parameters and seed.
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

## 7. Workspace Directory Structure

The data and model directories should be structured as follows:

```text
automl_workspace/
├── config/           # All config files
├── data_pipeline/
│   ├── input/        # Add your images here
│   ├── labeled/      # Labeled images and annotations
│   ├── augmented/    # Augmented images
│   ├── label_studio/ # Label Studio related files
│   └── ...
├── model_registry/
│   ├── model/        # Model weights
│   ├── distilled/    # Distilled model outputs
│   └── quantized/    # Quantized model outputs
└── master_dataset/   # Archived labeled datasets
```