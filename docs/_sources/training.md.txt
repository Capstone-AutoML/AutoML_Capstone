# Training

The training pipeline handles the fine-tuning of a YOLOv8 model on labeled images that have undergone the augmentation step. It ensures that correct model is selected, configures training parameters, runs the training process, and logs metadata and test performance.

## Steps
**1. Configuration Loading**: Loads a user-defined JSON configuration file (`train_config.json`) that specifies: 
- training hyperparameters (eg. `epoch`, `imgsz`), 
- Path to `data.yaml` file
- path to a base model to update

**2. Model Selection**: Identifies the latest updated YOLO model by searching through the `model_registry` directory and selecting the filename with the latest date name. The pattern of the file matches `*_updated_yolo.pt`. If no updated model exists, it falls back to a default `initial_model_path`.

**3. Training**: The model weights are updated by utilizing the custom training parameters from the config file and the selected base model. Training output is saved in a timestamped model name (eg. `2025-06-01_09_42_00_updated_yolo.pt`). Model performance data is saved under the `model_info/[model_name]/runs/` directory.

**4. Metadata Logging**: A `metadata.json` file is saved along with model weights including:
- Model name
- Timestamp
- Path to the original base model
- Training arguments
- Evaluation metrics

## Model Registry Structure
```
model_registry/
└── model/
    └── [updated_model_name].pt      # Trained weights
    └── model_info/
        └── [updated_model_name]/
            └── runs/                # YOLO training logs and results
            └── metadata.json        # Metadata file

```

## Training Configuration
The `train_config.json` file defines all parameters needed to run training. It includes:
- Training hyperparameters passed onto the YOLO model
- Path to the dataset (`data.yaml`)
- Fallback model to use if no updated model is found
- **Optional**: Path to a specific YOLO base the user prefers to train from, overriding both the fallback and latest model
- In-training augmentation parameters. 
    - By default the in-training augmentation is disabled (probabilities are set to 0.0). If user prefers to apply augmentation during training instead of as pre-processing step, the following parameters can be adjusted in the config:
```
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.0,
    "mixup": 0.0,
```
These values are passed directly into the YOLOv8 training loop and controls augmentation such as color, flipping, etc. Additional training arguments can be passed onto the `train_config.json` file as long as they are supported by YOLOv8 API. For more information on YOLOv8 training see [YOLOv8 documentation.](https://docs.ultralytics.com/modes/train/#train-settings)
