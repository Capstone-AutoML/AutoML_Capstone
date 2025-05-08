# AutoML_Capstone

## Wildfire Detection Data Pipeline

### Author: Sepehr Heydarian, Archer Liu, Elshaday Yoseph, Tien Nguyen

### Description

This project implements an intelligent, semi-automated data pipeline for improving a wildfire object detection model. The system is designed to continuously ingest unlabelled images, generate initial annotations using AI models, refine them through human-in-the-loop review, and retrain the base model. The pipeline also includes optimization steps (e.g. distillation and quantization) to prepare models for deployment on edge devices.

### Development Guidelines

- Use docstrings for all functions and classes
- Keep code modular and well-organized
- Use `typing` for type hints so that input and output types are clearly defined e.g.

  ```python
  from typing import Dict, List, Optional, Tuple

  def function(param: int) -> str:
    pass

  def function2(param: Dict) -> List[str]:
    pass
  ```

- Keep functions small and focused (single responsibility)
- Use descriptive variable and function names
- Use consistent naming conventions e.g. `labelling_yolo()`, `labelling_sam()`

#### Branch Naming Convention

To keep development organized, please follow this naming format for branches:

**Examples:**

- `labelling/yolo` – Work related to using YOLO for automated labelling  
- `labelling/SAM` – Work using Segment Anything Model (SAM) for labelling  
- `augmentation/data` – Work related to data augmentation
- `training/base` – Work related to training the base model  
- `training/distillation` – Work related to distillation  
- `training/quantization` – Work related to quantization  

Use dashes `-` to separate multiple words when needed (e.g. `labelling/segment-anything`).
