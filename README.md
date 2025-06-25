# AutoML_Capstone

## Wildfire Detection Model CI/CT/CD Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31112/)
[![Python code coverage testing with pytest](https://github.com/sep-he/AutoML_Capstone/actions/workflows/test.yml/badge.svg)](https://github.com/sep-he/AutoML_Capstone/actions/workflows/test.yml)

**Authors:** Sepehr Heydarian, Rongze(Archer) Liu, Elshaday Yoseph, Tien Nguyen

## Description

This project implements an intelligent, semi-automated data pipeline for improving a wildfire object detection model. The system is designed to continuously ingest unlabelled images, generate initial annotations using AI models, refine them through human-in-the-loop review, and retrain the base model. The pipeline also includes model compression steps (e.g. distillation and quantization) to prepare models for deployment on edge devices. For more technical details, refer to the [full documentation](https://capstone-automl.github.io/AutoML_Capstone/).

## Motivation

Manual labeling of wildfire imagery is time-consuming and error-prone. In addition, models degrade over time as environmental conditions and data distributions shift. Our system aims to continuously learn from new data using a scalable, semi-supervised approach. It automates as much of the machine learning workflow as possible and involves human review only when necessary.

## Key Features

- Automated pre-labeling using YOLOv8 and Grounding DINO
- Model matching and validation using IoU and confidence thresholds
- Human-in-the-loop review for mismatches via Label Studio
- Image augmentation to improve generalization
- End-to-end training, distillation, and quantization
- CI/CD/CT-compatible design for regular updates and retraining

## Setup Guide

This guide will walk you through setting up and running the AutoML CI/CD/CT: Continuous Training and Deployment Pipeline project.

### 1. Clone the Repository

```bash
git clone https://github.com/Capstone-AutoML/AutoML_Capstone.git
cd AutoML_Capstone
```

### 2. Configure Process Options

Control which pipeline steps to run via `pipeline_config.json`:

```text
// Set to true to skip a step
"process_options": {
  "skip_human_review": false,
  "skip_training": false,
  "skip_distillation": false,
  "skip_quantization": false
}
```

### 3. Run the Pipeline with Docker

**Important**: Docker cannot handle interactive Label Studio sessions for human review. Before running with Docker, you **must** disable human review in `automl_workspace/config/pipeline_config.json`:

```json
"process_options": {
  "skip_human_review": true
}
```

#### 💻 If You Have a GPU (CUDA Supported)

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

#### 💻 If You Have a CPU-Only Machine (No NVIDIA GPU)

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

### 4. Run Tests (Optional)

To verify the setup and run unit tests:

```bash
docker compose run test
```

---

### 5. Generate Bounding Box Visualizations (Optional)

To run the script that overlays bounding boxes on sample and labeled images using predictions from YOLO, DINO, and mismatched sources:

```bash
docker compose run generate_box
```

---
This will:

* Sample and draw 10 images each from YOLO, DINO, and mismatched directories.

* Draw bounding boxes on all images from the labeled directory.

- Save the visualized outputs under `automl_workspace/data_pipeline/boxed_images/`

### 6. Human Review with Label Studio

For human-in-the-loop validation using Label Studio, refer to the [Human Intervention](https://capstone-automl.github.io/AutoML_Capstone/human_in_loop.html) documentation.

## Future Development Guide

To continue development based on the current project setup, follow the steps below using the provided conda environment.

### Prerequisites

Before getting started, ensure you have the following installed:

- **Conda or Miniconda** - For environment management

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/Capstone-AutoML/AutoML_Capstone.git
cd AutoML_Capstone
```

**2. Set up environments:**

**For Full Pipeline (includes pre-labeling, training, distillation, and quantization):**

```bash
conda env create -f environment.yml
conda activate capstone_env

# Install GroundingDINO (required for full pipeline)
# To keep your workspace clean, it's recommended to clone the repository outside the main project directory
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install .
```

**For Human Review Only:**

```bash
conda env create -f human_review_env.yml
conda activate human_review_env
```

> **Note**: Both environments may be needed depending on your workflow. The human review is integrated in the main pipeline.

**3. GPU Support (Optional):**

```bash
# Activate the full pipeline environment
conda activate capstone_env

# Check CUDA version
nvcc -V

# Install GPU PyTorch (example for CUDA 12.4)
pip uninstall torch torchvision
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

> ⚠️ **Compatibility Note:** PyTorch 2.6.0 has known compatibility issues with GroundingDINO and Ultralytics. PyTorch 2.5.1 is recommended as shown above. If you need to use PyTorch 2.6.0 or higher, please refer to the [GroundingDINO issue](https://github.com/IDEA-Research/GroundingDINO/issues/405).

### Running the Pipeline

#### Configuration

Before running the pipeline, you can customize the behavior by modifying the configuration files in the `automl_workspace/config/` directory:

- **`pipeline_config.json`** - Main pipeline settings (thresholds, augmentation, distillation parameters)
- **`augmentation_config.json`** - Data augmentation settings (seed, number of augmentations, etc.)
- **`train_config.json`** - Model training configuration (epochs, batch size, learning rate, etc.)
- **`distillation_config.yaml`** - Distillation settings (model paths, epochs, patience, etc.)
- **`quantize_config.json`** - Model quantization settings (labeled images paths, quantization method, etc.)

> ⚠️ **Compatibility Note:** Due to ongoing compatibility issues between required packages (such as `imx500-converter`, `uni-pytorch`, and `model-compression-toolkit`), we are currently unable to support IMX quantization in this pipeline. The default option in `quantize_config.json` is `FP16`. If you require IMX quantization, you may need to experiment with manual package pinning or use a separate, isolated environment. Refer to [Sony IMX500 Export for Ultralytics YOLO11](https://docs.ultralytics.com/integrations/sony-imx500/) and [Raspberry Pi AI Camera IMX500 Converter User Manual](https://developer.aitrios.sony-semicon.com/en/raspberrypi-ai-camera/documentation/imx500-converter?version=3.14.3&progLang=) for future development.

#### Add Your Own Dataset

If you want to use your own dataset as input to the pipeline, create a folder structured as `automl_workspace/data_pipeline/input/` and place your images inside it.

The distillation dataset is a subset of labeled images, which is used to train the student model. It is a folder that contains the train images/labels and validation images/labels. The folder should have the following name & structure:

```txt
distillation_dataset/
   train/
     images/
     labels/
   val/
     images/
     labels/
```

It is currently assumed to be located in the `automl_workspace/data_pipeline/distillation` directory. When a new custom distillation dataset is provided, the user can overwrite the `distillation_dataset` attribute in `distillation_config.yaml` with the either relative or absolute path to the **directory** of the new custom distillation dataset.

#### Run the Full Pipeline in Conda Environment

```bash
python src/main.py
```

#### Run Human-in-the-Loop Review

```bash
python src/pipeline/human_intervention.py
```

#### Draw Boxed Images

```bash
python src/generate_boxed_images.py
```

## Support

Encountering issues? Need assistance? For any questions regarding this pipeline, please open an issue in the GitHub repository.
