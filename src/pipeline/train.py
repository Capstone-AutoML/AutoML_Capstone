from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import json

from pathlib import Path

def find_latest_model(model_dir: str, fallback_model: str) -> str:
    """
    Finds the YOLO model with the latest date in the filename.
    If none found, returns the fallback model.

    Args:
        model_dir (str): Directory containing YOLO model `.pt` files.
        fallback_model (str): Path to fallback model (used if none found).

    Returns:
        str: Path to the latest-dated model or the fallback model.
    """
    model_dir = Path(model_dir)
    models = sorted(
        model_dir.glob("*_updated_yolo.pt"),
        key=lambda x: x.stem,
        reverse=True
    )

    if models:
        return str(models[0])
    else:
        print(f"[WARN] Updated model not found. Falling back to: {fallback_model}")
        return fallback_model
    
def load_train_config(config_path: str) -> dict:
    """
    Loads training configuration from JSON file.
    Args:
        config_path (str): Path to the train_config.json file.
    Returns:
        dict: configuration dictionary
    """
    config_path = Path(config_path)
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Error: Config file not found at {config_path}")
    # Load JSON into dictionary
    with open(config_path, "r") as f:
        config = json.load(f)
    
    assert "training_config" in config, "Error: 'training_config' section missing in train_config.json"
    assert "data_yaml_path" in config, "Error: 'data_yaml_path' section missing in train_config.json"
    assert "initial_model_path" in config, "Error: 'initial_model_path' section missing in train_config.json"

    return config


def train_model(config: dict) -> str:
    """
    Trains a YOLOv8 model using the Ultralytics library and saves the trained model and metadata.

    Args:
        config (dict): Loaded config dictionary from train_config.json, containing:
            - data_yaml_path (str): Path to `data.yaml`
            - torch_device (str): 'cpu' or 'cuda'
            - training_config (dict): eg., epochs, lr0, imgsz, batch, workers, etc
            - model_path (str): (Optional) Path to a pre-trained model to fine-tune.

    Returns:
        str: Path to the saved trained model (.pt)
    """
    # Define paths
    model_dir = Path("automl_workspace/model_registry/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    user_model_path = config.get("model_path")
    initial_model_path = config.get("initial_model_path", "automl_workspace/model_registry/model/nano_trained_model.pt")

    if user_model_path:
        model_path = user_model_path
        print(f"[INFO] Using model specified in config: {model_path}")
    else:
        model_path = find_latest_model(model_dir, initial_model_path)
        print(f"[INFO] Using latest model: {model_path}")

    # Load YOLO model
    model = YOLO(model_path)

    # Extract training parameters
    train_args = config["training_config"]
    train_args["data"] = config["data_yaml_path"]
    train_args["device"] = config.get("torch_device", "cpu")

    # Generate a timestamped name if user did not specify one
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    output_model_name = config.get("output_model_name") or f"{timestamp}_updated_yolo.pt"
    trained_model_path = model_dir / output_model_name

    # Define metadata and runs output directory
    model_info_dir = model_dir / "model_info" / output_model_name.replace(".pt", "")
    model_info_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = model_info_dir / "metadata.json" 
    run_output_dir = model_info_dir / "runs" 
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Save model output
    train_args["project"] = str(model_info_dir)
    train_args["name"] = "train"

    # Run training with all arguments from config
    model.train(**train_args)
    
    # Save trained model
    model.save(str(trained_model_path))

    # Save metadata with training info
    metadata = {
        "model_name": output_model_name,
        "trained_from": model_path,
        "timestamp": timestamp,
        "training_args": train_args
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Evaluate on test set if defined in data.yaml
    try:
        test_results = model.val(split='test')
        test_metrics = {
            "map_50": test_results.box.map50,
            "map_75": test_results.box.map75,
            "map_50_95": test_results.box.map,
        }
        metadata["test_metrics"] = test_metrics
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
 
    except Exception as e:
        print(f"[WARN] Test evaluation failed or skipped: {e}")

    print(f"[INFO] Training complete. Model saved to {trained_model_path}")
    print(f"[INFO] Metadata saved to {metadata_path}")
    return str(trained_model_path)


# Entry point for standalone use
if __name__ == "__main__":
    config = load_train_config("train_config.json")
    train_model(config)