# Distillation

## Overview

The distillation process is a knowledge transfer technique that trains a smaller, more efficient model (student) to mimic the behavior of a larger, more complex model (teacher). This script implements model distillation specifically for wildfire detection using YOLOv8.

### Assumptions

- The teacher model is a YOLOv8 model that is pretrained & finetuned on a large dataset.
- The student model is a YOLOv8 model that is pretrained, and we will distill the teacher's knowledge into it during training & finetuning.
- The teacher model is a larger model than the student model.
- The student model is a smaller model than the teacher model.

### Inputs

- **Teacher Model**: A pre-trained YOLOv8 model that serves as the knowledge source
- **Student Model**: A smaller YOLOv8 model (YOLOv8n) that will be trained to mimic the teacher
- **Training Data**: Images and their corresponding annotations for wildfire detection (5 classes: FireBSI, LightningBSI, PersonBSI, SmokeBSI, VehicleBSI)
- **Configuration**: Training parameters defined in `student_model_cfg.yaml`

### Processing

The distillation process follows these main steps:

1. **Model Initialization**
   - Loads the pre-trained teacher model
   - Initializes the student model with **pretrained weights (on default COCO 80 classes dataset)**.
   - Configures model parameters and training settings

2. **Data Preparation**
   - Sets up training and validation datasets
   - Configures data loaders with appropriate batch sizes and augmentations
   - Automatically creates dataset directories and YAML configuration files

3. **Training Loop**
   - Implements knowledge distillation through a combination of:
     - Detection loss (for direct object detection learning)
     - Distillation loss (to mimic teacher's predictions)
   - Uses gradient clipping and learning rate scheduling
   - Supports checkpointing for training resumption
   - Includes validation during training with early stopping

### Outputs

- **Trained Student Model**: A compressed model that maintains detection performance
- **Training Logs**: Detailed metrics including:
  - Total loss
  - Bounding box loss
  - Classification loss
  - Distillation loss
  - Gradient norms
- **Checkpoints**: Model states saved at regular intervals
- **Validation Results**: Performance metrics on the validation dataset
- **Best Model**: Automatically saved model with highest validation fitness

### Key Features

- Supports layer freezing for transfer learning
- Implements both detection and distillation losses
- Provides comprehensive logging and checkpointing
- Includes validation during training with early stopping
- Supports training resumption from checkpoints
- Automatic best model tracking and restoration
- Debug mode for detailed loss component analysis
- NaN/Inf detection and handling for training stability
- Flexible logging at batch or epoch level

## Distillation Method and Hyperparameter Justification

### Method: Response-Based Distillation with Top-K Selection

This method distills the final outputs of the teacher model—bounding boxes and class confidence scores—into the student model. The key innovation is using **Top-K selection** instead of NMS filtering to identify the most confident teacher predictions for distillation.

**Why Top-K selection instead of NMS?**
  Top-K selection provides more stable and predictable distillation targets by selecting the K most confident predictions from the teacher, regardless of spatial overlap. This approach ensures consistent distillation signal and avoids the complexity of NMS parameter tuning.

**Why response-based?**
  This avoids needing to align intermediate representations, which is especially useful when teacher and student have different depths or backbones. Instead, we treat the teacher's predictions as refined pseudo-labels. This is a good baseline for distillation setup.

### Loss Weight Hyperparameters

```python
hyperparams = {
    "lambda_distillation": 2.0,
    "lambda_detection": 1.0,
    "lambda_dist_ciou": 1.0,
    "lambda_dist_kl": 2.0,
    "temperature": 2.0
}
```

#### Explanation of Each Weight

1. **`lambda_distillation = 2.0`**

   -This global weight amplifies the entire distillation loss relative to the standard detection loss.
   -Since the goal is to make the student mimic the teacher, this slightly higher weight promotes more attention on distillation without overpowering ground-truth learning.

2. **`lambda_detection = 1.0`**

   -This ensures the student still respects ground-truth labels.
   -Helps avoid cases where the teacher may be confidently wrong (e.g., due to domain shift or noisy training).

3. **`lambda_dist_ciou = 1.0`**

   -This balances the bounding box alignment with the classification component.
   -CIoU (Complete IoU) already provides strong geometric supervision; no need to overweight it unless box alignment is especially poor.

4. **`lambda_dist_kl = 2.0`**

   -A higher weight helps capture the teacher's soft class probabilities, which encode "dark knowledge" (i.e., relative confidence between classes).
   -Especially important for class imbalance scenarios or rare classes.

5. **`temperature = 2.0`**

   -Controls the softness of class distributions during distillation.
   -A moderate temperature like 2.0 makes logits softer and gradients smoother—helping the student learn inter-class relationships more effectively.

## Training Configuration

These training settings are defined in `student_model_cfg.yaml` and chosen to ensure stable, effective knowledge transfer:

| Parameter      | Value          | Reason                                                                        |
| -- | -- | -- |
| `imgsz`        | 640            | Balanced choice for stability and memory usage                                |
| `lr0`          | 0.005          | Lower than YOLO default to slow learning, to compensate for potentially more unstable gradient in distillation |
| `batch`        | 32             | Balanced choice for stability and memory usage                                |
| `epochs`       | 200            | Allows enough time for full knowledge transfer                                |
| Early Stopping | 100 epochs     | Prevents unnecessary overfitting if student plateaus                          |
| Optimizer      | SGD + momentum | Well-tested in for default YOLOv8, works well for distillation settings                               |
| LR Scheduler   | LambdaLR   | Helps avoid local minima and promotes smooth convergence                      |
| Grad Clipping  | 10.0           | Prevents exploding gradients, improves training stability                     |

## Distillation Deep Dive

### Training Loop Summary

Each training step includes:

1. Forward pass of student on batch images.
2. Forward pass of teacher (in `eval` mode) to get stable predictions.
3. **Top-K selection** of teacher predictions based on confidence scores.
4. Matching student and teacher predictions using the selected top-K indices.
5. Computing the loss:
   -Detection loss using YOLOv8's native `v8DetectionLoss`
   -Distillation loss with CIoU (for box) and KL divergence (for class), using softened logits.
6. Combining both using weighted sum and backpropagating.
7. **Validation step** after each epoch with fitness calculation.
8. **Best model tracking** and early stopping based on validation fitness.

### Loss Components

-**Detection Loss (YOLO native)**
  -CIoU for box regression
  -BCE for classification
  -Distribution Focal Loss (DFL) for box refinement

-**Distillation Loss**
  -**Box:** CIoU between student and teacher predictions (top-K selected)
  -**Class:** KL divergence between softened logits (student vs. teacher)
  -Combined via `lambda_distillation * (λ_ciou * ciou_loss + λ_kl * kl_loss)`

```python
total_loss = (
    lambda_detection * detection_loss +
    lambda_distillation * (
        lambda_dist_ciou * box_distillation_loss +
        lambda_dist_kl * cls_distillation_loss
    )
)
```

### Top-K Selection Process

The distillation loss computation follows these steps:

1. **Extract teacher confidence scores** from class predictions
2. **Select top-K predictions** per image based on confidence
3. **Gather corresponding student predictions** using the same indices
4. **Compute losses** on the matched prediction pairs
5. **Apply temperature scaling** for KL divergence computation

This approach ensures that distillation focuses on the teacher's most confident predictions, providing cleaner learning signals to the student.

## Model Architecture Considerations

-**Student Model:** YOLOv8n (lightweight and fast)
-**Teacher Model:** Larger YOLOv8 variant (e.g., `m`, `l`, or `x`)
-**Freezing:** You may freeze early layers of the student backbone to focus learning on the head. This is because the backbone is already pretrained features that are useful for the student to learn from.
-**Anchor points, feature map resolution:** Kept consistent between student and teacher for compatibility
-**Class count:** Configured for 5 wildfire-specific classes instead of COCO's 80 classes

## Training Stability Features

-**Gradient Clipping (10.0):** Prevents instability from large gradients
-**Monitoring for NaNs/Infs:** Training loop skips if numerical instability is detected
-**Loss logging per batch/epoch:** Helps isolate spikes or anomalies in distillation loss
-**Temperature scaling:** Avoids overly confident logits that could destabilize KL divergence
-**Model copying for validation:** Prevents training mode conflicts during validation
-**Best model state tracking:** Maintains the best performing model throughout training

## Final Remarks

While this setup represents a well-reasoned and empirically grounded starting point for response-based distillation in YOLOv8, it's important to recognize that distillation is inherently iterative. The balance between detection and distillation losses, temperature scaling, gradient stability, and optimizer configuration often requires substantial trial and error, especially when adapting to different datasets or shifting between teacher and student architectures. Nevertheless, this configuration provides a strong initial baseline that captures key principles of effective knowledge transfer. As the system matures, it can be further refined through advanced techniques such as feature-based distillation, dynamic loss weighting, teacher ensemble methods, or self-training with pseudo-labeling, depending on the application domain and available resources.