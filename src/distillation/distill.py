# Python standard library
import json
import os
import sys
from importlib import reload
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
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torchvision.ops.ciou_loss import complete_box_iou_loss
import torch.nn.functional as F
from tqdm import tqdm

# Add parent directory to path
sys.path.append("..")

# Ultralytics imports
from ultralytics import YOLO
from ultralytics.models.yolo.model import DetectionModel
from ultralytics.cfg import get_cfg, YAML
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.data.build import build_yolo_dataset, build_dataloader, YOLODataset, InfiniteDataLoader
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils.tal import make_anchors
from ultralytics.utils.ops import non_max_suppression, xywh2xyxy
from ultralytics.utils.torch_utils import one_cycle

# Custom modules
from utils import load_config, detect_device

pipeline_config = load_config("../pipeline_config.json")
device = pipeline_config["torch_device"] if pipeline_config["torch_device"] else detect_device()
print(f"Using device: {device}")

def load_models(device: str, base_dir: Path) -> Tuple[YOLO, YOLO]:
    """
    Load teacher and student models.
    
    Args:
        device: Device to load models on
        base_dir: Base directory for model paths
        
    Returns:
        Tuple of (teacher_yolo, student_yolo) models
    """
    # Load the teacher model (our pre-trained model)
    teacher_yolo = YOLO(
        base_dir / 'mock_io/model_registry/model/nano_trained_model.pt', 
    ).to(device)
    
    # Load the student model (our new model, random initialized weights)
    student_yolo = YOLO("yolov8n-5class.yaml").load("yolov8n.pt").to(device) 
    student_yolo.yaml["nc"] = 5
    student_model = student_yolo.model
    student_model.nc = 5
    
    # Load config
    config_dict = YAML.load("student_model_cfg.yaml")
    student_model.args = get_cfg(config_dict)
    
    # Sanity check
    assert student_yolo.nc == 5, "student_yolo.nc should be 5"
    assert student_model.nc == 5, "student_model.nc should be 5"
    assert isinstance(teacher_yolo, nn.Module)
    assert isinstance(student_yolo, nn.Module)
    
    return teacher_yolo, student_yolo

def prepare_dataset(img_path: Path, student_model: nn.Module, batch_size: int = 16, mode: str = "train") -> Tuple[YOLODataset, DataLoader]:
    """
    Prepare dataset and dataloader for training.
    
    Args:
        img_path: Directory containing images
        batch_size: Batch size for training
        
    Returns:
        Tuple of (dataset, dataloader)
        
    Note: 
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
            kldivloss(s_cls_log_softmax, t_cls_log_softmax, log_target=True) * 
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
    lambdas: Dict[str, float] = {
        "lambda_distillation": 2.0,
        "lambda_detection": 1.0,
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0
    }
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
        "distillation_loss": np.array([])
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
                hyperparams=lambdas
            ).to(device)
            
            # Check for NaN in individual loss components
            if torch.isnan(bbox_loss) or torch.isnan(cls_loss) or torch.isnan(dfl_loss) or torch.isnan(distillation_loss):
                print(f"NaN detected in loss components at batch {batch_idx}:")
                print(f"bbox_loss: {bbox_loss}, cls_loss: {cls_loss}, dfl_loss: {dfl_loss}, distillation_loss: {distillation_loss}")
                continue
            
            # Calculate total loss
            total_loss = (
                lambdas["lambda_detection"] * detection_losses.sum() + 
                lambdas["lambda_distillation"] * distillation_loss + 
                lambdas["lambda_dist_ciou"] * bbox_loss + 
                lambdas["lambda_dist_kl"] * cls_loss
            )
            
            # Final NaN check before backward pass
            if torch.isnan(total_loss).any() or torch.isinf(total_loss).any():
                print(f"NaN detected in total loss at batch {batch_idx}")
                print(f"Component losses: bbox={bbox_loss}, cls={cls_loss}, dfl={dfl_loss}, dist={distillation_loss}")
                continue
                
            total_loss.backward()
            
            # Clip gradients to prevent exploding gradients
            clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Store losses
            batch_loss_dict["bbox_loss"] = np.append(batch_loss_dict["bbox_loss"], bbox_loss)
            batch_loss_dict["cls_loss"] = np.append(batch_loss_dict["cls_loss"], cls_loss)
            batch_loss_dict["dfl_loss"] = np.append(batch_loss_dict["dfl_loss"], dfl_loss)
            batch_loss_dict["distillation_loss"] = np.append(
                batch_loss_dict["distillation_loss"], 
                distillation_loss.cpu().detach().numpy()
            )
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {str(e)}")
            continue
    
    return batch_loss_dict


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    student_model: nn.Module,
    optimizer: optim.Optimizer,
    losses: Dict[str, float],
) -> None:
    """
    Save model checkpoint.
    
    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
        student_model: Student model to save
        optimizer: Optimizer state to save
        losses: Dictionary of loss values
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': student_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
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
    }
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
        detection_criterion: Detection loss criterion
        config_dict: Configuration dictionary
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        save_checkpoint_every: Save checkpoint every n epochs
        detection_validator: Optional validator for validation after training
        
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
                hyperparams=hyperparams
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
    }
) -> Dict[str, List[float]]:
    """
    Start the distillation training process.
    
    Args:
        device: Device to train on
        base_dir: Base directory for paths
        save_checkpoint_every: Save checkpoint every n epochs
        frozen_layers: Number of layers to freeze in the backbone
        lambdas: Dictionary of lambda values for each loss function

    Returns:
        Dictionary containing lists of loss values for each epoch
    """
    # Load models
    teacher_yolo, student_yolo = load_models(device, base_dir)
    teacher_model = teacher_yolo.model
    student_model = student_yolo.model
    
    model_args = student_model.args
    model_args.mode = "train"

    BATCH_SIZE = model_args.batch
    EPOCHS = model_args.epochs

    # Freeze backbone layers if specified
    if frozen_layers > 0:
        freeze_layers(student_model, frozen_layers)
    
    # Prepare dataset
    train_dataset, train_dataloader = prepare_dataset(img_path=base_dir / img_dir / "train", student_model=student_model, batch_size=BATCH_SIZE, mode="train")
    val_dataset, val_dataloader = prepare_dataset(img_path=base_dir / img_dir / "valid", student_model=student_model, batch_size=BATCH_SIZE, mode="val")
    
    # Setup training
    detection_trainer = DetectionTrainer(cfg=model_args)
    
    optimizer, learning_rate_scheduler = build_optimizer_and_scheduler(
        model=student_model,
        detection_trainer=detection_trainer,
        model_args=model_args
    )
    
    detection_criterion = v8DetectionLoss(model=student_model)
    
    # Create checkpoint directory
    checkpoint_dir = Path("./checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
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
        config_dict=student_model.args,
        device=device,
        checkpoint_dir=checkpoint_dir,
        save_checkpoint_every=save_checkpoint_every,
        hyperparams=hyperparams
    )
    
    # # Validation
    model_args.mode = "val"
    detection_validator = DetectionValidator(dataloader=val_dataloader, args=model_args)
    detection_validator(model=student_model)

if __name__ == "__main__":
    
    hyperparams = {
        "lambda_distillation": 2.0,
        "lambda_detection": 1.0,
        "lambda_dist_ciou": 1.0,
        "lambda_dist_kl": 2.0,
        "temperature": 2.0
    }
    start_distillation(
        device=device, 
        base_dir=Path("..", ".."), 
        img_dir=Path("mock_io/data/distillation/distillation_dataset"),
        frozen_layers=10,
        save_checkpoint_every=25,
        hyperparams=hyperparams
    )