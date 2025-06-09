# AutoML_Capstone

## Wildfire Detection Model CI/CT/CD Pipeline

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31112/)
[![Python code coverage testing with pytest](https://github.com/sep-he/AutoML_Capstone/actions/workflows/test.yml/badge.svg)](https://github.com/sep-he/AutoML_Capstone/actions/workflows/test.yml)

**Authors:** Sepehr Heydarian, Rongze(Archer) Liu, Elshaday Yoseph, Tien Nguyen

## Description

This project implements an intelligent, semi-automated data pipeline for improving a wildfire object detection model. The system is designed to continuously ingest unlabelled images, generate initial annotations using AI models, refine them through human-in-the-loop review, and retrain the base model. The pipeline also includes optimization steps (e.g. distillation and quantization) to prepare models for deployment on edge devices.

## Motivation

Manual labeling of wildfire imagery is time-consuming and error-prone. In addition, models degrade over time as environmental conditions and data distributions shift. Our system aims to continuously learn from new data using a scalable, semi-supervised approach. It automates as much of the machine learning workflow as possible and involves human review only when necessary.

## Key Features

- Automated pre-labeling using YOLOv8 and Grounding DINO
- Model matching and validation using IoU and confidence thresholds
- Human-in-the-loop review for mismatches via Label Studio
- Image augmentation to improve generalization
- End-to-end training, distillation, and quantization
- CI/CD/CT-compatible design for regular updates and retraining

## Quick Start with Docker

If you want to quickly run the full pipeline(excluding the human-in-the-loop), you can use [Docker](https://docs.docker.com/get-docker/) for an easy setup. Docker is **highly recommended** for this project as it simplifies the environment setup and ensures all dependencies are correctly installed.

```bash
docker compose up
```

For the human-in-the-loop review, please refer to the Usage section below.

## Usage

### Prerequisites

Before getting started, ensure you have the following installed:

- **Python 3.11**
- **Conda or Miniconda** - For environment management

> **Note**: The pipeline can run entirely on CPU, but GPU acceleration significantly improves performance for model training and distillation.

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/Capstone-AutoML/AutoML_Capstone.git
cd AutoML_Capstone
```

**2. Set up environments:**

**For Full Pipeline (includes pre-labeling, training, quantization):**

```bash
conda env create -f environment.yml
conda activate capstone_env

# Install GroundingDINO (required for full pipeline)
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install .
cd ..
```

**For Human Review Only:**

```bash
conda env create -f human_review_env.yml
conda activate human_review_env
```

> **Note**: Both environments may be needed depending on your workflow. The human review interface runs independently from the main pipeline.

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

Before running the pipeline, you can customize the behavior by modifying the configuration files in the `src/` directory:

- **`pipeline_config.json`** - Main pipeline settings (thresholds, augmentation, distillation parameters)
- **`train_config.json`** - Model training configuration (epochs, batch size, learning rate, etc.)
- **`quantize_config.json`** - Model quantization settings (labeled images paths, quantization method, etc.)

If you want to use GPU for the pipeline, set `"torch_device": "cuda"` in both `pipeline_config.json` and `train_config.json`:

#### Run the Full Pipeline in Conda Environment

```bash
python src/main.py
```

#### Run Human-in-the-Loop Review

```bash
python src/pipeline/human_review.py
```

#### Draw Boxed Images

```bash
python src/generate_boxed_images.py
```

## Support

Encountering issues? Need assistance? For any questions regarding this pipeline, please open an issue in the GitHub repository.
