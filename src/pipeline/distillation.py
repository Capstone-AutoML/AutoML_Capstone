"""
Script for model optimization through distillation.

This module implements knowledge distillation for YOLOv8 models, where a smaller student model
learns from a larger teacher model. The distillation process combines standard detection loss
with distillation loss to transfer knowledge effectively.
"""
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO
import torch.nn as nn
import torch.nn.functional as F
import yaml
import json
from datetime import datetime
from tqdm import tqdm
from ultralytics.utils.loss import DFLoss, VarifocalLoss, FocalLoss, BboxLoss, bbox_iou
from ultralytics.utils.ops import non_max_suppression
import cv2
import os


class CIoULoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target):
        return bbox_iou(pred, target, x1y1x2y2=True, CIoU=True)

class YOLODataset(torch.utils.data.Dataset):
    """
    Dataset class for YOLOv8 model training.
    
    This class provides a PyTorch dataset for training YOLOv8 models. It handles
    image loading, resizing, and annotation parsing. The dataset supports optional
    data augmentation and normalization transformations.
    It will yield images and their corresponding ground truth boxes
    in the format of [class_label, x_center, y_center, width, height]
    Note that x_center, y_center, width, height are normalized to the image size, thus are float 0-1
    """
    def __init__(self, img_dir, label_dir, img_size=640, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size  # can be int or (H, W) tuple
        self.transforms = transforms
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = Path(self.images[idx])
        img_path = Path(self.img_dir, img_name)
        label_path = Path(self.label_dir, img_name.stem + ".txt")

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        if isinstance(self.img_size, int):
            target_h, target_w = self.img_size, self.img_size
        else:
            target_h, target_w = self.img_size
        # https://github.com/ultralytics/ultralytics/issues/4510#issuecomment-1689938511
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Load and scale annotations
        boxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                class_label, x, y, w, h = map(float, line.strip().split())
                boxes.append([class_label, x, y, w, h])

        # permute image to (C, H, W) and normalize pixel values to 0-1
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        print(img_path)
        print(label_path)
        return image, torch.tensor(boxes)

def _save_distill_config(config: Dict, config_registry_path: Path) -> str:
    """
    Save distillation configuration to the config registry.
    
    Args:
        config (Dict): Distillation configuration to save
        config_registry_path (Path): Path to config registry directory
        
    Returns:
        str: Path to saved config file
    """
    # Create timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = config_registry_path / f"distill_config_{timestamp}.json"
    
    # Add metadata
    config_with_metadata = {
        "metadata": {
            "timestamp": timestamp,
            "type": "distillation",
            "version": "1.0"
        },
        "config": config
    }
    
    # Save to file
    with open(config_path, 'w') as f:
        json.dump(config_with_metadata, f, indent=2)
    
    return str(config_path)

def _compute_detection_loss(
    student_preds: torch.Tensor,
    ground_truth: torch.Tensor,
    config: Dict
) -> torch.Tensor:
    """
    See https://github.com/ultralytics/ultralytics/blob/3556ef31fbd43da044fb6765e8e26e2698b2cbdc/ultralytics/utils/loss.py#L194
    for the implementation of the loss functions
    Compute the standard detection loss between student predictions and ground truth.
    
    This function calculates the YOLO detection loss using:
    - BCEWithLogitsLoss for classification
    - CIoU Loss for box regression
    - Distribution Focal Loss for precise box localization
    
    Args:
        student_preds (torch.Tensor): Predictions from the student model, containing
            objectness scores, class probabilities, and bounding box coordinates
        ground_truth (torch.Tensor): Ground truth labels in YOLO format, containing
            objectness, class labels, and bounding box coordinates
        config (Dict): Configuration containing loss parameters such as:
            - box_loss_weight: Weight for bounding box regression loss
            - cls_loss_weight: Weight for classification loss
            - dfl_loss_weight: Weight for distribution focal loss
            
    Returns:
        torch.Tensor: Total detection loss value
    """
    # Initialize loss components
    loss_cls = nn.BCEWithLogitsLoss(reduction='none')
    loss_box = CIoULoss(reduction='mean')
    loss_dfl = DFLoss()
    
    # Extract predictions and ground truth
    pred_cls = student_preds[..., 4:]  # Class predictions
    pred_box = student_preds[..., :4]  # Box coordinates
    pred_dfl = student_preds[..., 4:8]  # DFL predictions
    
    gt_cls = ground_truth[..., 4:]  # Ground truth classes
    gt_box = ground_truth[..., :4]  # Ground truth boxes
    gt_dfl = ground_truth[..., 4:8]  # Ground truth DFL targets
    
    # Compute losses
    cls_loss = loss_cls(pred_cls, gt_cls)
    box_loss = loss_box(pred_box, gt_box)
    dfl_loss = loss_dfl(pred_dfl, gt_dfl)
    
    # Get weights from config
    box_weight = config.get('box_loss_weight', 7.5)
    cls_weight = config.get('cls_loss_weight', 0.5)
    dfl_weight = config.get('dfl_loss_weight', 1.5)
    
    # Combine losses
    total_loss = (box_weight * box_loss + 
                 cls_weight * cls_loss + 
                 dfl_weight * dfl_loss)
    
    return total_loss

def _compute_distillation_loss(
    student_preds: torch.Tensor,
    teacher_preds: torch.Tensor,
    temperature: float,
    config: Dict
) -> torch.Tensor:
    """
    Compute the distillation loss between student and teacher predictions.
    
    This function implements knowledge distillation using KL divergence between
    the softened probability distributions of teacher and student models. The
    temperature parameter controls the softness of the probability distributions.
    
    Args:
        student_preds (torch.Tensor): Raw logits from the student model
        teacher_preds (torch.Tensor): Raw logits from the teacher model
        temperature (float): Temperature parameter for softening probabilities.
            Higher values make the distribution softer
        config (Dict): Configuration containing:
            - distillation_loss_weight: Weight for the distillation loss
            - feature_distillation: Whether to include feature-level distillation
            
    Returns:
        torch.Tensor: Distillation loss value
    """
    pass

def _combine_losses(
    detection_loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    Combine detection and distillation losses using weighted sum.
    
    This function implements the weighted combination of the standard detection
    loss and the distillation loss. The alpha parameter controls the trade-off
    between learning from ground truth and learning from the teacher model.
    
    Args:
        detection_loss (torch.Tensor): Standard detection loss between student
            predictions and ground truth
        distillation_loss (torch.Tensor): Distillation loss between student
            and teacher predictions
        alpha (float): Weight for detection loss (1-alpha for distillation).
            Should be between 0 and 1
            
    Returns:
        torch.Tensor: Combined loss value
    """
    pass

def _train_distillation_step(
    student_model: YOLO,
    teacher_model: YOLO,
    batch: torch.Tensor,
    ground_truth: torch.Tensor,
    config: Dict
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Perform a single training step for knowledge distillation.
    
    This function executes one training iteration of the distillation process,
    including forward passes through both models, loss computation, and
    backpropagation. It handles the teacher model in evaluation mode and
    the student model in training mode.
    
    Args:
        student_model (YOLO): Student model to train, in training mode
        teacher_model (YOLO): Teacher model for guidance, in evaluation mode
        batch (torch.Tensor): Batch of input images
        ground_truth (torch.Tensor): Ground truth labels for the batch
        config (Dict): Training configuration containing:
            - temperature: For softening probabilities
            - alpha: Weight for detection vs distillation loss
            - optimizer: Optimizer configuration
            - learning_rate: Learning rate for training
            
    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: Combined loss and dictionary
            containing individual loss components (detection_loss, distillation_loss)
    """
    pass

def _get_dataloader(
    img_dir: Path, label_dir: Path, batch_size: int=16,shuffle: bool=True
) -> DataLoader:
    
    def yolo_collate_fn(batch):
        images, targets = zip(*batch)
        return list(images), list(targets)

    dataset = YOLODataset(
        img_dir=img_dir,
        label_dir=label_dir,
        transforms=None
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=yolo_collate_fn
    )
    return data_loader

def distill_model(
    teacher_model_path: str,
    train_data: str,
    config: Dict,
    output_dir: Path
) -> Tuple[str, Dict]:
    """
    Main function to perform model distillation.
    
    This function orchestrates the entire distillation process, including:
    - Loading teacher and student models
    - Setting up data loaders
    - Training loop with distillation
    - Model checkpointing
    - Performance evaluation
    
    Args:
        teacher_model_path (str): Path to the teacher model weights
        student_model_path (str): Path to the student model weights
        train_data (str): Path to training data configuration (YAML)
        config (Dict): Distillation configuration containing:
            - epochs: Number of training epochs
            - batch_size: Batch size for training
            - temperature: For softening probabilities
            - alpha: Weight for detection vs distillation loss
            - optimizer: Optimizer configuration
            - learning_rate: Learning rate for training
            - device: Device to use for training
        output_dir (Path): Directory to save the distilled model and logs
        
    Returns:
        Tuple[str, Dict]: Path to distilled model and dictionary containing
            training metrics (loss history, validation performance, etc.)
    """
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Load models
    teacher_model = YOLO(teacher_model_path).model
    student_model = YOLO("yolov8n.yaml").model
    
    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    teacher_model.to(device)
    student_model.to(device)
    
    # Set models to appropriate modes
    teacher_model.eval()
    student_model.train()
    
    # Get data loader
    data_loader = _get_dataloader(
        img_dir=train_data,
        label_dir=train_data,
        batch_size=config.get("batch_size", 16),
        shuffle=config.get("shuffle", True)
    )
    
    # Training loop
    for epoch in range(config.get("epochs", 100)):
        for batch in data_loader:
            imgs, targets = batch
            pass
    
    # Initialize metrics dictionary
    metrics = {
        "detection_loss": [],
        "distillation_loss": [],
        "total_loss": [],
        "val_metrics": []
    }
    
    
