# Distillation

This page outlines the unit test coverage for the `distillation.py` module, which handles knowledge distillation between YOLO teacher and student models. The tests focus on validating loss computation, feature decoding, gradient monitoring, and layer freezing utilities.

---

## Coverage Overview

This test suite includes coverage for:

- Loading model components and datasets
- Calculating distillation loss
- Decoding head features into bounding boxes
- Computing gradient norms
- Freezing model layers

---

## Fixtures

### `test_config`  
Loads a distillation-specific YAML configuration file from `src/distillation_config.yaml`.

### `mock_teacher_model`  
Returns a mock convolutional teacher model using PyTorch `nn.Sequential`.

### `mock_student_model`  
Returns a simplified convolutional student model for testing distillation and gradient behavior.

---

## Function Tests

### `calculate_gradient_norm`

- `test_calculate_gradient_norm`:  
  Computes the L2 norm of gradients after a backward pass on a dummy model. Asserts the value is positive and a float.

---

### `freeze_layers`

- `test_freeze_layers`:  
  Confirms that specified layers in the model are frozen (i.e., `requires_grad = False`) while others remain trainable.

---

### `head_features_decoder`

- `test_head_features_decoder`:  
  Passes dummy head feature maps and simulates decoding into bounding boxes using a mocked detection criterion. Ensures the result is a tensor of the expected shape.

---

### `compute_distillation_loss`

- `test_compute_distillation_loss`:  
  Computes the loss between teacher and student predictions with matching shape and format. Confirms the result is a scalar tensor greater than zero.

---

## Summary

This page outlines a focused test suite that ensures:

- Distillation loss calculations are stable and positive  
- Bounding box decoding works across feature map scales  
- Model layers can be frozen selectively for fine-tuning  
- Gradient norms are measurable during training  

These tests support reliable implementation of student-teacher training in the AutoML pipeline.
