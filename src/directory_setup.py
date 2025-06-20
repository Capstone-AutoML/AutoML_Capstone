import os

def create_automl_workspace(base_path="."):
    """
    Create the main directory structure for an AutoML pipeline project.

    This function creates the following structure under the specified base path:
    
        automl_workspace/
        ├── data_pipeline/     - For data required for pipeline processes: labeling, human intervention, augmenting, fine-tuning.
        ├── model_registry/    - For storing trained models and related information
        ├── master_dataset/    - For storing all images and labels that have been labeled by the pipeline, and database tables
        └── config/            - Configuration files (YAML/JSON) for pipeline steps

    If any of the directories already exist, they will not be recreated.

    Args:
        base_path (str): The base directory where 'automl_workspace' should be created.
                         Defaults to the current working directory.

    Returns:
        None
    """
    root_dir = os.path.join(base_path, "automl_workspace")

    subdirs = [
        "data_pipeline/input",
        "data_pipeline/prelabeled",
        "data_pipeline/labeled",
        "data_pipeline/augmented",
        "data_pipeline/training",
        "data_pipeline/distillation",
        "data_pipeline/quantization",
        "data_pipeline/label_studio/pending",
        "data_pipeline/label_studio/tasks",
        "data_pipeline/label_studio/results",
        "model_registry/model",
        "model_registry/distilled",
        "model_registry/quantized",
        "master_dataset",
        "config"
    ]

    for subdir in subdirs:
        path = os.path.join(root_dir, subdir)
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            print(f"Error creating directory {path}: {e}")


# Only run if this script is executed directly
if __name__ == "__main__":
    create_automl_workspace()