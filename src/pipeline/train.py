from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import json

def find_latest_model(model_dir: str, fallback_model: str) -> str:
    """
    Finds the most recently modified YOLO model in the specified directory.
    If no model is found, returns the fallback model path.

    Args:
        model_dir (str): Directory containing YOLO model `.pt` files.
        fallback_model (str): Path to fallback model (used if none found).

    Returns:
        str: Path to the most recent model or the fallback model.   
    """
    model_dir = Path(model_dir)
    models = sorted(model_dir.glob("*.pt"), key=lambda x: x.stat().st_mtime, reverse=True)

    if models:
        return str(models[0])
    else:
        print(f"Updated model not found. Falling back to: {fallback_model}")
        return fallback_model
    
def train_model(config: dict) -> str:
    """
    Trains a YOLOv8 model using the Ultralytics library and saves the trained model and metadata.

    Workflow:
    - Loads most recent model from registry or falls back to `nano_trained_model.pt`
    - Uses training parameters from `config['training_config']` in `pipeline_config.json`
    - Saves trained model with timestamp-based name
    - Saves metadata about training

    Args:
        config (dict): Pipeline configuration containing:
            - data_yaml_path (str): Path to `data.yaml`
            - torch_device (str): 'cpu' or 'cuda'
            - training_config (dict): with keys: epochs, lr0, imgsz, batch, workers

    Returns:
        str: Path to the saved trained model (.pt)
    """
    # Define paths
    model_dir = Path("mock_io/model_registry/model")
    metadata_dir = model_dir / "model_metadata"
    model_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Load training configuration
    train_config = config.get("training_config", {})
    device = config.get("torch_device", "cpu")
    data_yaml = config["data_yaml_path"]

    # Training hyperparameters
    epochs = train_config.get("epochs", 100)
    lr0 = train_config.get("lr0", 0.001)
    imgsz = train_config.get("imgsz", 640)
    batch = train_config.get("batch", 16)
    workers = train_config.get("workers", 8)

    # Load model: latest available or fallback base model
    initial_model_path = str(model_dir / "nano_trained_model.pt")
    model_path = config.get("model_path", find_latest_model(model_dir, initial_model_path)) # model_path can be specified in `pipeline_config.json`
    model = YOLO(model_path)

    print(f"Training from: {model_path}")
    print(f"Epochs: {epochs} | LR: {lr0} | ImgSize: {imgsz} | Batch: {batch}")

    # Training the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        lr0=lr0,
        imgsz=imgsz,
        batch=batch,
        workers=workers,
        device=device
    ) 

    # Save model with timestamped name
    timestamp = datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    model_name = f"{timestamp}_updated_yolo.pt"
    new_model_path = model_dir / model_name
    model.save(str(new_model_path))

    # Save metadata
    metadata = {
        "model_name": model_name,
        "date": timestamp,
        "trained_on": data_yaml,
        "epochs": epochs,
        "lr0": lr0,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "source_model": model_path
    }

    metadata_path = metadata_dir / f"{model_name.replace('.pt', '_metadata.json')}"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to: {new_model_path}")
    print(f"Metadata saved to: {metadata_path}")
    return str(new_model_path)