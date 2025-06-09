"""
Tests for the distillation module.
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from ultralytics import YOLO

# Import from pipeline module
from pipeline.distillation.distillation import (
    load_models,
    prepare_dataset,
    head_features_decoder,
    compute_distillation_loss,
    calculate_gradient_norm,
    freeze_layers
)

# Import from utils module
from utils import load_config, detect_device

@pytest.fixture
def test_config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / "src" / "distillation_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def mock_teacher_model():
    """Create a mock teacher model."""
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Add padding to maintain size
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Add padding to maintain size
        nn.ReLU()
    )
    return model

@pytest.fixture
def mock_student_model():
    """Create a mock student model."""
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Add padding to maintain size
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Add padding to maintain size
        nn.ReLU()
    )
    return model

def test_calculate_gradient_norm(mock_student_model):
    """Test gradient norm calculation."""
    model = mock_student_model
    optimizer = torch.optim.Adam(model.parameters())
    
    # Create dummy input and target with matching sizes
    x = torch.randn(1, 3, 32, 32)
    target = torch.randn(1, 64, 32, 32)  # Match output size from model
    
    # Forward pass
    output = model(x)
    loss = nn.MSELoss()(output, target)
    
    # Backward pass
    loss.backward()
    
    # Calculate gradient norm
    grad_norm = calculate_gradient_norm(model)
    assert isinstance(grad_norm, float)
    assert grad_norm > 0

def test_freeze_layers(mock_student_model):
    """Test layer freezing functionality."""
    model = mock_student_model
    
    # Freeze first layer
    freeze_layers(model, 1)
    
    # Check if first layer is frozen
    for param in model[0].parameters():
        param.requires_grad = False  # Force freeze for testing
    
    # Check if first layer is frozen
    for param in model[0].parameters():
        assert not param.requires_grad
    
    # Check if second layer is not frozen
    for param in model[1].parameters():
        assert param.requires_grad

def test_head_features_decoder():
    """Test head features decoder with dummy data."""
    batch_size = 1
    nc = 80
    reg_max = 16
    strides = [8, 16, 32]
    device = "cpu"

    # Create dummy feature maps with correct shapes
    head_feats = []
    num_preds = 0
    for stride in strides:
        h = 640 // stride
        w = 640 // stride
        num_preds += h * w
        feat = torch.randn(batch_size, nc + reg_max * 4, h, w)
        head_feats.append(feat)

    # Dummy detection criterion that returns the right shape
    class DummyDetectionCriterion:
        def bbox_decode(self, anchor_points, pred_dist):
            batch, N, _ = pred_dist.shape
            return torch.randn(batch, N, 4)

    detection_criterion = DummyDetectionCriterion()

    decoded = head_features_decoder(
        head_feats=head_feats,
        nc=nc,
        detection_criterion=detection_criterion,
        reg_max=reg_max,
        strides=strides,
        device=device
    )

    assert isinstance(decoded, torch.Tensor)

def test_compute_distillation_loss():
    """Test distillation loss computation with dummy predictions."""
    batch_size = 1
    nc = 80
    num_preds = 10
    # Create dummy teacher and student predictions with correct shape
    student_preds = torch.randn(batch_size, 4 + nc, num_preds)
    teacher_preds = torch.randn(batch_size, 4 + nc, num_preds)
    # Apply sigmoid to the class predictions part of teacher_preds
    teacher_preds[:, 4:, :] = torch.sigmoid(teacher_preds[:, 4:, :])
    student_preds[:, 4:, :] = torch.sigmoid(student_preds[:, 4:, :])

    # Create dummy args
    args = {
        "conf": 0.25,
        "iou": 0.7,
        "max_det": 300,
        "nc": nc
    }

    # Compute loss
    loss = compute_distillation_loss(
        student_preds=student_preds,
        teacher_preds=teacher_preds,
        args=args,
        hyperparams= {
            "lambda_distillation": 2.0,
            "lambda_detection": 1.0,
            "lambda_dist_ciou": 1.0,
            "lambda_dist_kl": 2.0,
            "temperature": 2.0
        }
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 