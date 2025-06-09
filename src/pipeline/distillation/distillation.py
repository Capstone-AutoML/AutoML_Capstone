# Python standard library
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Literal
# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torchvision.ops.ciou_loss import complete_box_iou_loss
import torch.nn.functional as F
from tqdm import tqdm
import csv
from datetime import datetime

# Add parent directory to path
sys.path.append("..")

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.models.yolo.model import DetectionModel
from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.data.build import build_yolo_dataset, build_dataloader, YOLODataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import non_max_suppression
from ultralytics.utils.torch_utils import one_cycle

# Custom modules
from utils import load_config, detect_device

SCRIPT_DIR = Path(__file__).parent

def load_models(device: str, base_dir: Path, distillation_config: Dict[str, Any]) -> Tuple[YOLO, YOLO]:
    """
    Load teacher and student models.
    
    Args:
        device: Device to load models on
        base_dir: Base directory for model paths
        distillation_config: Configuration dictionary for distillation
        
    Returns:
        Tuple of (teacher_yolo, student_yolo) models
    """
    # Load the teacher model (our pre-trained model)
    teacher_yolo = YOLO(
        distillation_config["teacher_model"], 
    ).to(device)
    
    # Load the student model (our new model, random initialized weights)
    student_yolo = (
        YOLO(base_dir / "pipeline/distillation/yolov8n-5class.yaml")
        .load(base_dir / "pipeline/distillation/yolov8n.pt")
    ).to(device) 
    student_yolo.yaml["nc"] = 5
    student_model = student_yolo.model
    student_model.nc = 5
    
    # Set model args from distillation config
    student_model.args = get_cfg(distillation_config)
    
    # Sanity check
    assert student_yolo.nc == 5, "student_yolo.nc should be 5"
    assert student_model.nc == 5, "student_model.nc should be 5"
    assert isinstance(teacher_yolo, nn.Module)
    assert isinstance(student_yolo, nn.Module)
    
    return teacher_yolo, student_yolo

def prepare_dataset(img_path: Path, student_model: nn.Module, batch_size: int = 16, mode: str = "train") -> Tuple[YOLODataset, DataLoader]:
    """
    Prepare dataset and dataloader for training.
    
    Notes: 
        number_of_objects_detected: the number of objects detected in all images in the batch
        batch_size: number of images in the batch
    
    - each batch in the train_dataloader contains:
    - batch_idx:
        tensor of shape (number_of_objects_detected), 
        for each object, the value is 0, ... batch_size - 1, 
        depending on the index of the image that the object belongs to in the batch
    - img: image tensor of shape (batch_size, 3, 640, 640)
    - bboxes: bboxes tensor of shape (number_of_objects_detected, 4),  4 is for normalized x1, y1, x2, y2
    - cls: cls tensor of shape (number_of_objects_detected, 1), containing all class labels of the objects detected in the batch
    - resized_shape: Resized 2D dim of the image. A list of tensor, first tensor is first dim, second tensor is second dim
    - ori_shape: Original 2D dim of the image. Alist of tensor, first tensor is first dim, second tensor is second dim
    
    Args:
        img_path: Directory containing images
        student_model: Student model instance
        batch_size: Batch size for training
        mode: Dataset mode ("train" or "val")
        
    Returns:
        Tuple of (dataset, dataloader)
    """
    data = {
        "names": {
            0: "FireBSI", 
            1: "LightningBSI", 
            2: "PersonBSI", 
            3: "SmokeBSI", 
            4: "VehicleBSI"
        },
        "channels": 3,
    }
    
    train_dataset = build_yolo_dataset(
        cfg=student_model.args,
        img_path = img_path,
        batch=batch_size,
        data=data,
        mode=mode,
    )
    
    train_dataloader = build_dataloader(
        train_dataset, 
        batch=batch_size, 
        workers=0,
        shuffle=False, 
    )
    
    return train_dataset, train_dataloader

def head_features_decoder(
    head_feats: List[torch.Tensor], 
    nc: int, 
    detection_criterion: v8DetectionLoss, 
    reg_max: int = 16, 
    strides: List[int] = [8, 16, 32], 
    device: str = "cpu"
) -> torch.Tensor:
    """
    Decode the head features into bounding boxes and class scores.
    
    Args:
        head_feats: List of tensors, each representing a feature map from a detection head
        nc: Number of classes
        detection_criterion: Detection loss criterion
        reg_max: Maximum number of bounding box parameters
        strides: List of strides for the feature maps
        device: Device to perform computations on
        
    Returns:
        Tensor: pred_concatted: Concatenated bounding boxes and class raw logits scores
                Shape is (batch_size, 4 + num_classes, total_predictions)
    """
    b = head_feats[0].shape[0] # batch size
    dfl_vals = reg_max * 4 # number of dfl encoded channels for bounding boxes
    no = nc + dfl_vals # number of out channels
    
    pred_dist, pred_scores = torch.cat(
        [feat.view(b, no, -1) for feat in head_feats], dim=2
    ).split(
        (dfl_vals, nc), dim=1
    )
    
    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_dist = pred_dist.permute(0, 2, 1).contiguous()

    anchor_points, _ = make_anchors(head_feats, strides, 0.5)
    anchor_points = anchor_points.to(device)
    pred_bboxes = detection_criterion.bbox_decode(anchor_points, pred_dist)
    assert torch.any(pred_scores < 0) or torch.any(pred_scores > 1), "pred_scores should be logits, not sigmoid"
    pred_concatted = torch.permute(torch.cat((pred_bboxes, pred_scores), dim=2), (0, 2, 1))
    
    return pred_concatted

def compute_distillation_loss(
    student_preds: torch.Tensor, 
    teacher_preds: torch.Tensor, 
    args: Dict[str, Any], 
    nc: int = 80, 
    device: str = "cpu", 
    eps: float = 1e-7,
    reduction: Literal["batchmean", "sum"] = "batchmean",
    hyperparams: Dict[str, float] = {
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0
    }
) -> torch.Tensor:
    """
    Compute the distillation loss between the student and teacher predictions.
    
    Args:
        student_preds: The student predictions
        teacher_preds: The teacher predictions
        args: Configuration arguments
        nc: Number of classes
        device: Device to perform computations on
        eps: Small epsilon value for numerical stability
        reduction: Reduction method for the loss ("batchmean" or "sum")
        hyperparams: Dictionary of hyperparameters for loss functions
        
    Returns:
        Total distillation loss
    """
    if not isinstance(student_preds, torch.Tensor) or not isinstance(teacher_preds, torch.Tensor):
        raise ValueError("student_preds and teacher_preds must be tensors")

    batch_size = student_preds.shape[0]
    dtype = teacher_preds.dtype
    kldivloss = nn.KLDivLoss(reduction=reduction, log_target=True)
    # eps = torch.finfo().eps
    eps = 1e-7

    # Split the concatenated bounding boxes and class scores
    s_bbox, s_cls_logits = torch.split(student_preds, (4, nc), dim=1)
    s_cls_sigmoid = torch.sigmoid(s_cls_logits)
    assert torch.all(s_cls_sigmoid >= 0) and torch.all(s_cls_sigmoid <= 1), "s_cls_sigmoid should be sigmoid, not logits"
    teacher_preds_full = teacher_preds.clone()
    assert torch.all(teacher_preds_full[:, 4:, :] >= 0) and torch.all(teacher_preds_full[:, 4:, :] <= 1), "teacher_preds_full should be sigmoid, not logits"
    student_preds_full = torch.cat((s_bbox, s_cls_sigmoid), dim=1)
    
    common_nms_args = {
        "conf_thres": args.get("conf", 0.25) if args.get("conf") else 0.25,
        "iou_thres": args.get("iou", 0.7) if args.get("iou") else 0.7,
        "classes": args.get("classes", None),
        "agnostic": args.get("agnostic_nms", False) if args.get("agnostic_nms") else False,
        "max_det": args.get("max_det", 300) if args else 300,
        "nc": 0,
        "return_idxs": True,
        "max_time_img": 1
    }

    _, teacher_preds_final_idxs = non_max_suppression(
        prediction=teacher_preds_full,
        **common_nms_args,
    )
    
    selected_student_raw_predictions_list = []
    selected_teacher_raw_predictions_list = []

    for i in range(batch_size):
        student_preds_for_image_i = student_preds_full[i, ...].transpose(0, 1)
        teacher_preds_for_image_i = teacher_preds_full[i, ...].transpose(0, 1)
        indices_to_select = teacher_preds_final_idxs[i]
        
        if indices_to_select.numel() > 0:
            selected_student_preds = student_preds_for_image_i[indices_to_select]
            selected_teacher_preds = teacher_preds_for_image_i[indices_to_select]
            
            selected_student_raw_predictions_list.append(selected_student_preds)
            selected_teacher_raw_predictions_list.append(selected_teacher_preds)

    # get the actual batch size (batches with results)
    actual_batch_size = len(selected_student_raw_predictions_list)
    batch_box_regression_loss = torch.zeros(actual_batch_size, dtype=dtype)
    batch_cls_loss = torch.zeros(actual_batch_size, dtype=dtype)
    
    for i in range(actual_batch_size):
            
        tp, sp = selected_teacher_raw_predictions_list[i], selected_student_raw_predictions_list[i]
        s_bboxes, s_cls_sigmoid = torch.split(sp, (4, nc), dim=1)
        t_bboxes, t_cls_sigmoid = torch.split(tp, (4, nc), dim=1)

        ciou_loss = complete_box_iou_loss(s_bboxes, t_bboxes, reduction="mean")
        
        s_cls_logit = torch.logit(s_cls_sigmoid, eps=eps)
        s_cls_log_softmax = F.log_softmax(s_cls_logit / hyperparams["temperature"], dim=1)
        t_cls_logit = torch.logit(t_cls_sigmoid, eps=eps)
        t_cls_log_softmax = F.log_softmax(t_cls_logit / hyperparams["temperature"], dim=1)
        kl_div_loss = (
            kldivloss(s_cls_log_softmax, t_cls_log_softmax) * 
            (hyperparams["temperature"]**2)
        )
        
        batch_box_regression_loss[i] = ciou_loss
        batch_cls_loss[i] = kl_div_loss
    
    if reduction == "batchmean":
        total_loss = (
            hyperparams["lambda_dist_ciou"] * batch_box_regression_loss.mean() + 
            hyperparams["lambda_dist_kl"] * batch_cls_loss.mean()
        )
    else:
        total_loss = (
            hyperparams["lambda_dist_ciou"] * batch_box_regression_loss.sum() + 
            hyperparams["lambda_dist_kl"] * batch_cls_loss.sum()
        )

    return total_loss

def calculate_gradient_norm(model: nn.Module) -> float:
    """
    Calculate the total gradient norm across all parameters.
    
    Args:
        model: The model to calculate gradient norm for
        
    Returns:
        Total gradient norm as a float
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def log_training_metrics(
    log_file: Path,
    epoch: int,
    batch_idx: Optional[int],
    losses: Dict[str, float],
    grad_norm_before: Optional[float] = None,
    grad_norm_after: Optional[float] = None,
    is_new_file: bool = False,
    log_level: Literal["batch", "epoch"] = "epoch"
) -> None:
    """
    Log training metrics to a CSV file.
    
    Args:
        log_file: Path to the log file
        epoch: Current epoch number
        batch_idx: Current batch index (None for epoch-level logging)
        losses: Dictionary of loss values
        grad_norm_before: Gradient norm before clipping
        grad_norm_after: Gradient norm after clipping
        is_new_file: Whether this is the first write to the file
        log_level: Whether to log at batch or epoch level
    """
    fieldnames = [
        'timestamp', 'epoch', 'batch', 
        'total_loss', 'bbox_loss', 'cls_loss', 'dfl_loss', 'dist_loss',
        'grad_norm_before', 'grad_norm_after'
    ]
    
    # Prepare the row data
    row = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'epoch': epoch,
        'batch': batch_idx if log_level == "batch" else "epoch",
        'total_loss': losses.get('total_loss', ''),
        'bbox_loss': losses.get('bbox_loss', ''),
        'cls_loss': losses.get('cls_loss', ''),
        'dfl_loss': losses.get('dfl_loss', ''),
        'dist_loss': losses.get('dist_loss', ''),
        'grad_norm_before': grad_norm_before if grad_norm_before is not None else '',
        'grad_norm_after': grad_norm_after if grad_norm_after is not None else ''
    }
    
    # Write to file
    mode = 'w' if is_new_file else 'a'
    with open(log_file, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)

def train_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_dataloader: DataLoader,
    detection_trainer: DetectionTrainer,
    optimizer: optim.Optimizer,
    detection_criterion: v8DetectionLoss,
    config_dict: Dict[str, Any],
    device: str = "cpu",
    nc: int = 5,
    hyperparams: Dict[str, float] = {
        "lambda_distillation": 2.0,
        "lambda_detection": 1.0,
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0
    },
    epoch: int = 1,
    log_file: Optional[Path] = None,
    log_level: Literal["batch", "epoch"] = "batch",
    debug: bool = False
) -> Dict[str, float]:
    """
    Train for one epoch.
    """
    student_model.train()
    teacher_model.eval()
    
    batch_loss_dict = {
        "total_loss": np.array([]),
        "bbox_loss": np.array([]),
        "cls_loss": np.array([]),
        "dfl_loss": np.array([]),
        "distillation_loss": np.array([]),
        "grad_norm_before": np.array([]),
        "grad_norm_after": np.array([])
    }
    
    for batch_idx, batch in enumerate(train_dataloader):
            
        optimizer.zero_grad()
        
        try:
            preprocessed_batch = detection_trainer.preprocess_batch(batch)
            inputs = preprocessed_batch["img"].to(device)
            targets = preprocessed_batch["cls"].to(device)
            
            # Additional validation after preprocessing
            if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                print(f"NaN/Inf detected in preprocessed images at batch {batch_idx}")
                continue
                
            if torch.isnan(targets).any() or torch.isinf(targets).any():
                print(f"NaN/Inf detected in preprocessed targets at batch {batch_idx}")
                continue
            
            student_head_feats = student_model(inputs)
            detection_losses, detection_losses_detached = detection_criterion(preds=student_head_feats, batch=batch) 
            bbox_loss, cls_loss, dfl_loss = detection_losses_detached.cpu()

            with torch.no_grad():
                teacher_inputs = batch["img"].to(device)
                teacher_preds, _ = teacher_model(teacher_inputs)
            
            student_preds = head_features_decoder(
                head_feats=student_head_feats, 
                nc=nc, 
                detection_criterion=detection_criterion, 
                device=device
            ).to(device)
            
            distillation_loss = compute_distillation_loss(
                student_preds, 
                teacher_preds,
                config_dict, 
                nc=nc, 
                device=device,
                reduction="batchmean",
                hyperparams=hyperparams
            ).to(device)
            
            # Calculate total loss with proper scaling and type conversion
            detection_loss = detection_losses.sum()
            bbox_loss = bbox_loss.to(device)
            cls_loss = cls_loss.to(device)
            dfl_loss = dfl_loss.to(device)
            
            # Debug print individual losses
            if debug:
                print(f"\nBatch {batch_idx} Loss Components:")
                print(f"Detection Loss: {detection_loss.item():.4f}")
                print(f"Bbox Loss: {bbox_loss.item():.4f}")
                print(f"Cls Loss: {cls_loss.item():.4f}")
                print(f"DFL Loss: {dfl_loss.item():.4f}")
                print(f"Distillation Loss: {distillation_loss.item():.4f}")
                
            # Calculate weighted components
            weighted_detection = hyperparams["lambda_detection"] * detection_loss
            weighted_dist = hyperparams["lambda_distillation"] * distillation_loss
            
            # Debug print weighted components
            if debug:
                print(f"\nWeighted Components:")
                print(f"Weighted Detection: {weighted_detection.item():.4f}")
                print(f"Weighted Dist: {weighted_dist.item():.4f}")
            
            # Calculate total loss
            total_loss = weighted_detection + weighted_dist
            
            if debug:
                print(f"Total Loss: {total_loss.item():.4f}")
            
            # Final NaN check before backward pass
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print(f"NaN detected in total loss at batch {batch_idx}")
                print(f"Component losses: bbox={bbox_loss}, cls={cls_loss}, dfl={dfl_loss}, dist={distillation_loss}")
                continue
                
            total_loss.backward()
            
            # Calculate gradient norm before clipping
            grad_norm_before = calculate_gradient_norm(student_model)
            
            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(student_model.parameters(), max_norm=10.0)
            
            # Calculate gradient norm after clipping
            grad_norm_after = calculate_gradient_norm(student_model)
            
            optimizer.step()
            
            # Store losses and gradient norms
            batch_loss_dict["bbox_loss"] = np.append(batch_loss_dict["bbox_loss"], bbox_loss.cpu().detach().numpy())
            batch_loss_dict["cls_loss"] = np.append(batch_loss_dict["cls_loss"], cls_loss.cpu().detach().numpy())
            batch_loss_dict["dfl_loss"] = np.append(batch_loss_dict["dfl_loss"], dfl_loss.cpu().detach().numpy())
            batch_loss_dict["distillation_loss"] = np.append(
                batch_loss_dict["distillation_loss"], 
                distillation_loss.cpu().detach().numpy()
            )
            batch_loss_dict["total_loss"] = np.append(batch_loss_dict["total_loss"], total_loss.cpu().detach().numpy())
            batch_loss_dict["grad_norm_before"] = np.append(batch_loss_dict["grad_norm_before"], grad_norm_before)
            batch_loss_dict["grad_norm_after"] = np.append(batch_loss_dict["grad_norm_after"], grad_norm_after)
            
            # Log metrics if log_file is provided and log_level is batch
            if log_file is not None and log_level == "batch":
                current_losses = {
                    'total_loss': float(total_loss.cpu().detach().numpy()),
                    'bbox_loss': float(bbox_loss.cpu().detach().numpy()),
                    'cls_loss': float(cls_loss.cpu().detach().numpy()),
                    'dfl_loss': float(dfl_loss.cpu().detach().numpy()),
                    'dist_loss': float(distillation_loss.cpu().detach().numpy())
                }
                log_training_metrics(
                    log_file=log_file,
                    epoch=epoch,
                    batch_idx=batch_idx,
                    losses=current_losses,
                    grad_norm_before=grad_norm_before,
                    grad_norm_after=grad_norm_after,
                    is_new_file=(epoch == 1 and batch_idx == 0),
                    log_level=log_level
                )
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    
    # Log epoch-level metrics if log_level is epoch
    if log_file is not None and log_level == "epoch":
        epoch_losses = {
            'total_loss': float(np.mean(batch_loss_dict["total_loss"])),
            'bbox_loss': float(np.mean(batch_loss_dict["bbox_loss"])),
            'cls_loss': float(np.mean(batch_loss_dict["cls_loss"])),
            'dfl_loss': float(np.mean(batch_loss_dict["dfl_loss"])),
            'dist_loss': float(np.mean(batch_loss_dict["distillation_loss"]))
        }
        log_training_metrics(
            log_file=log_file,
            epoch=epoch,
            batch_idx=None,
            losses=epoch_losses,
            grad_norm_before=float(np.mean(batch_loss_dict["grad_norm_before"])),
            grad_norm_after=float(np.mean(batch_loss_dict["grad_norm_after"])),
            is_new_file=(epoch == 1),
            log_level=log_level
        )
    
    return batch_loss_dict


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    student_model: nn.Module,
    optimizer: optim.Optimizer,
    learning_rate_scheduler: LambdaLR,
    losses: Dict[str, float],
) -> None:
    """
    Save model checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
        student_model: Student model to save
        optimizer: Optimizer state to save
        learning_rate_scheduler: Learning rate scheduler state to save
        losses: Dictionary of loss values
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': student_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': learning_rate_scheduler.state_dict(),
        'loss': losses['total_loss'],
        'bbox_loss': losses['bbox_loss'],
        'cls_loss': losses['cls_loss'],
        'dfl_loss': losses['dfl_loss']
    }
    torch.save(checkpoint, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')


def freeze_layers(model: nn.Module, num_layers: int = 10) -> None:
    """
    Freeze the first n layers of the model.
    For example, if num_layers = 10, the first 10 layers (The Backbone) will be frozen.
    https://community.ultralytics.com/t/guidance-on-freezing-layers-for-yolov8x-seg-transfer-learning/189/2
    https://github.com/ultralytics/ultralytics/blob/3e669d53067ff1ed97e0dad0a4063b156f66686d/ultralytics/engine/trainer.py#L258
    
    Args:
        model: The model to freeze layers in
        num_layers: Number of layers to freeze from the start
    """
    # Get all parameters
    params = list(model.parameters())
    
    # Freeze the first n layers
    manual_freeze = [f"model.{i}." for i in range(num_layers)]
    perm_freeze = ['.dfl.']
    total_freeze = set(manual_freeze + perm_freeze)
    for k, v in model.named_parameters():
        for layer in total_freeze:
            if layer in k:
                v.requires_grad = False
    
    # Print which layers are frozen
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    total_count = sum(1 for p in model.parameters())
    print(f"Frozen {frozen_count}/{total_count} layers in the model")


def train_loop(
    num_epochs: int,
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_dataloader: DataLoader,
    detection_trainer: DetectionTrainer,
    optimizer: optim.Optimizer,
    learning_rate_scheduler: LambdaLR,
    detection_criterion: v8DetectionLoss,
    config_dict: Dict[str, Any],
    device: str,
    checkpoint_dir: Path,
    save_checkpoint_every: int,
    hyperparams: Dict[str, float] = {
        "lambda_distillation": 2.0,
        "lambda_detection": 1.0,
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0
    },
    start_epoch: int = 1,
    log_file: Optional[Path] = None,
    log_level: Literal["batch", "epoch"] = "epoch",
    debug: bool = False
) -> Dict[str, List[float]]:
    """
    Execute the complete training process including all epochs.
    
    Args:
        num_epochs: Number of epochs to train
        student_model: Student model to train
        teacher_model: Teacher model for distillation
        train_dataloader: DataLoader for training data
        detection_trainer: Detection trainer instance
        optimizer: Optimizer for training
        learning_rate_scheduler: Learning rate scheduler
        detection_criterion: Detection loss criterion
        config_dict: Configuration dictionary
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        save_checkpoint_every: Save checkpoint every n epochs
        hyperparams: Dictionary of hyperparameters for loss functions
        start_epoch: Start training from this epoch
        log_file: Optional path to log file for metrics
        log_level: Whether to log at batch or epoch level
        
    Returns:
        Dictionary containing lists of loss values for each epoch
    """
    epoch_losses = {
        'total_loss': [],
        'bbox_loss': [],
        'cls_loss': [],
        'dfl_loss': [],
        'dist_loss': []
    }
    
    try:
        for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs", position=0):
            # Train one epoch
            batch_loss_dict = train_epoch(
                student_model=student_model,
                teacher_model=teacher_model,
                train_dataloader=train_dataloader,
                detection_trainer=detection_trainer,
                optimizer=optimizer,
                detection_criterion=detection_criterion,
                config_dict=config_dict,
                device=device,
                hyperparams=hyperparams,
                epoch=epoch,
                log_file=log_file,
                log_level=log_level,
                debug=debug
            )
            
            learning_rate_scheduler.step()
            
            # Calculate average losses
            batch_loss_bbox = np.mean(batch_loss_dict["bbox_loss"]).round(4)
            batch_loss_cls = np.mean(batch_loss_dict["cls_loss"]).round(4)
            batch_loss_dfl = np.mean(batch_loss_dict["dfl_loss"]).round(4)
            batch_loss_dist = np.mean(batch_loss_dict["distillation_loss"]).round(4)
            batch_loss_total = batch_loss_bbox + batch_loss_cls + batch_loss_dfl + batch_loss_dist
            
            # Store losses
            epoch_losses['total_loss'].append(batch_loss_total)
            epoch_losses['bbox_loss'].append(batch_loss_bbox)
            epoch_losses['cls_loss'].append(batch_loss_cls)
            epoch_losses['dfl_loss'].append(batch_loss_dfl)
            epoch_losses['dist_loss'].append(batch_loss_dist)
            
            print(
                f"Epoch {epoch}: (Overall: {batch_loss_total}, bbox_loss: {batch_loss_bbox}, "
                f"cls_loss: {batch_loss_cls}, dfl_loss: {batch_loss_dfl}, dist_loss: {batch_loss_dist})"
            )
            
            # Save checkpoint
            if save_checkpoint_every > 0 and epoch % save_checkpoint_every == 0:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    student_model=student_model,
                    optimizer=optimizer,
                    learning_rate_scheduler=learning_rate_scheduler,
                    losses={
                        'total_loss': batch_loss_total,
                        'bbox_loss': batch_loss_bbox,
                        'cls_loss': batch_loss_cls,
                        'dfl_loss': batch_loss_dfl
                    }
                )
            
    except ValueError as e:
        print(str(e))
        print("Exit training, please check the training process again...")
    
    return epoch_losses


def build_optimizer_and_scheduler(
    model: DetectionModel,
    detection_trainer: DetectionTrainer,
    model_args: Dict[str, Any]
) -> Tuple[optim.Optimizer, LambdaLR]:
    """
    Build the optimizer and learning rate scheduler.
    
    Args:
        model: DetectionModel instance
        detection_trainer: DetectionTrainer instance
        model_args: Model arguments

    Returns:
        Tuple of optimizer and learning rate scheduler
    """
    
    optimizer = detection_trainer.build_optimizer(
        model=model,
        name=model_args.optimizer,
        lr=model_args.lr0,
        momentum=model_args.momentum,
        decay=model_args.weight_decay,
    )
    
    # https://github.com/ultralytics/ultralytics/blob/487e27639595047cff8775dab5e2ff268d8647c4/ultralytics/engine/trainer.py#L229
    if model_args.cos_lr:
        lambda_func = one_cycle(1, model_args.lrf, model_args.epochs)
    else:
        lambda_func = lambda x: max(1 - x / model_args.epochs, 0) * (1.0 - model_args.lrf) + model_args.lrf
    
    learning_rate_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda_func,
    )
    
    return optimizer, learning_rate_scheduler


def load_checkpoint(
    checkpoint_path: Path,
    student_model: nn.Module,
    optimizer: optim.Optimizer,
    learning_rate_scheduler: LambdaLR
) -> int:
    """
    Load a checkpoint and restore model and optimizer state.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        student_model: Student model to restore state to
        optimizer: Optimizer to restore state to
        learning_rate_scheduler: Learning rate scheduler to restore state to
        
    Returns:
        The epoch number from the checkpoint
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore model state
    student_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Restore optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore learning rate scheduler state if it exists
    if 'scheduler_state_dict' in checkpoint:
        learning_rate_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint['epoch']


def start_distillation(
    device: str = "cpu",
    base_dir: Path = Path(".."),
    img_dir: Path = Path("dataset"),
    save_checkpoint_every: int = 25,
    frozen_layers: int = 10, # freeze the Backbone layers
    hyperparams: Dict[str, float] = {
        "lambda_distillation": 2.0,
        "lambda_detection": 1.0,
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0
    },
    resume_checkpoint: Optional[Path] = None,
    output_dir: Path = Path("distillation_out"),
    log_level: Literal["batch", "epoch"] = "batch",
    debug: bool = False,
    distillation_config: Optional[Dict[str, Any]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None
) -> Dict[str, List[float]]:
    """
    Start the distillation training process.
    
    Args:
        device: Device to train on
        base_dir: Base directory for paths (should be SCRIPT_DIR from main.py)
        img_dir: Directory containing training images
        save_checkpoint_every: Save checkpoint every n epochs
        frozen_layers: Number of layers to freeze in the backbone
        hyperparams: Dictionary of hyperparameters for loss functions
        resume_checkpoint: Optional path to checkpoint to resume training from
        output_dir: Directory to save output
        log_level: Whether to log at batch or epoch level
        debug: Whether to print debug information
        distillation_config: Configuration dictionary for distillation
        pipeline_config: Configuration dictionary for pipeline
    Returns:
        Dictionary containing lists of loss values for each epoch
    """
    if distillation_config is None:
        raise ValueError("distillation_config is required")
        
    # Create output directory structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_dir / output_dir / timestamp
    logs_dir = output_dir / "logs"
    checkpoint_dir = output_dir / "checkpoints"
    
    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Create log file
    log_file = logs_dir / f'training_log_{timestamp}.csv'
    
    # Load models
    teacher_yolo, student_yolo = load_models(device, base_dir, distillation_config)
    teacher_model = teacher_yolo.model
    student_model = student_yolo.model
    
    # Use distillation config for model args
    model_args = get_cfg(distillation_config)
    model_args.mode = "train"

    BATCH_SIZE = model_args.batch
    EPOCHS = model_args.epochs

    # Freeze backbone layers if specified
    if frozen_layers > 0:
        freeze_layers(student_model, frozen_layers)
    
    # Prepare dataset
    train_dataset, train_dataloader = prepare_dataset(
        img_path=img_dir / "train", 
        student_model=student_model, 
        batch_size=BATCH_SIZE, 
        mode="train"
    )
    
    # Setup training
    detection_trainer = DetectionTrainer(
        cfg=model_args, 
        overrides={"data": Path(distillation_config["distillation_dataset"]) / "distillation_data.yaml"}
    )
    
    optimizer, learning_rate_scheduler = build_optimizer_and_scheduler(
        model=student_model,
        detection_trainer=detection_trainer,
        model_args=model_args
    )
    
    detection_criterion = v8DetectionLoss(model=student_model)
    
    # Load checkpoint if specified
    start_epoch = 1
    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        start_epoch = load_checkpoint(
            checkpoint_path=resume_checkpoint,
            student_model=student_model,
            optimizer=optimizer,
            learning_rate_scheduler=learning_rate_scheduler
        ) + 1  # Start from next epoch
        print(f"Resuming training and distillation from epoch {start_epoch}")
    else:
        print("Starting training and distillation from scratch")
    
    # Run training loop
    train_loop(
        num_epochs=EPOCHS,
        student_model=student_model,
        teacher_model=teacher_model,
        train_dataloader=train_dataloader,
        detection_trainer=detection_trainer,
        optimizer=optimizer,
        learning_rate_scheduler=learning_rate_scheduler,
        detection_criterion=detection_criterion,
        config_dict=model_args,
        device=device,
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_every=save_checkpoint_every,
        hyperparams=hyperparams,
        start_epoch=start_epoch,
        log_file=log_file,
        log_level=log_level,
        debug=debug
    )


if __name__ == "__main__":
    # Load configurations
    base_dir = Path(__file__).parent.parent.parent
    distillation_config = YAML.load(base_dir / "distillation_config.yaml")
    
    hyperparams = {
        "lambda_distillation": 2.0,
        "lambda_detection": 1.0,
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0,
        "temperature": 2.0
    }
    
    start_distillation(
        device=detect_device(),
        base_dir=base_dir,
        img_dir=Path("mock_io/data/distillation"),
        frozen_layers=10,
        save_checkpoint_every=25,
        hyperparams=hyperparams,
        resume_checkpoint=None,
        output_dir=Path(SCRIPT_DIR, "distillation_out"),
        log_level="batch",
        debug=False,
        distillation_config=distillation_config
    )