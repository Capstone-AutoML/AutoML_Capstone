# Setup Guide

This guide will walk you through setting up and running the AutoML CI/CD/CT: Continuous Training and Deployment Pipeline project.

## 1. Clone the Repository

First, clone the GitHub repository and navigate into the project directory:

```bash
git clone https://github.com/Capstone-AutoML/AutoML_Capstone.git
cd AutoML_Capstone
```

## 2. Run the Pipeline

Launch the entire pipeline using Docker Compose:

```bash
docker compose up
```

This command will:

* Download the necessary datasets and model files (only on first run, unless the `mock_io/data/sampled_dataset/` or `mock_io/model_registry/model` folders are removed.)
* Initialize and run the full automated pipeline.

## 3. Run Tests (Optional)

To verify the setup and run test scripts:

```bash
docker compose run test
```

## 4. GPU Support (Optional)

By default, the pipeline runs on **CPU** for maximum compatibility. If your machine supports GPU acceleration, you can switch to **CUDA** by editing the following configuration files:

* `train_config.json`
* `pipeline_config.json`

### Change the following key in each file:

```json
"torch_device": "cpu"
```

to:

```json
"torch_device": "cuda"
```

This will enable the pipeline to use your available GPU for faster training and inference.

## 5. Configuration Files

These are the two main configuration files used in the pipeline:

* `train_config.json`: Controls training parameters like batch size, learning rate, model paths, and training image details.
* `pipeline_config.json`: Controls distillation, augmentation, and matching thresholds.

Each file includes several tunable parameters for fine control of the pipeline behavior. For most users, the default values are sufficient.

---

You're now ready to use the AutoML pipeline. For more detailed information, refer to the individual module documentation sections.
