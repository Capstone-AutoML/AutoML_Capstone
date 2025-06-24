# Distillation Tests

This document describes the test suite for the distillation module, which is responsible for knowledge distillation in the wildfire detection pipeline. Knowledge distillation is a technique where a smaller, more efficient model (student) learns from a larger, more accurate model (teacher) to improve performance while reducing computational requirements.

## Overview

The distillation tests verify the functionality of the knowledge distillation pipeline, which is a critical component for model optimization in the wildfire detection system. The tests ensure that the distillation process works correctly for transferring knowledge from a pre-trained teacher model to a smaller student model.

## Test Structure

The test file `tests/test_distillation.py` contains the following test components:

### Fixtures

#### `test_config`
- **Purpose**: Loads test configuration from the distillation config file
- **Location**: `src/distillation_config.yaml`
- **Usage**: Provides configuration parameters for testing distillation components

#### `mock_teacher_model`
- **Purpose**: Creates a mock teacher model for testing
- **Architecture**: Simple CNN with 2 convolutional layers (3→64→128 channels)
- **Usage**: Simulates a pre-trained teacher model without loading actual weights

#### `mock_student_model`
- **Purpose**: Creates a mock student model for testing
- **Architecture**: Smaller CNN with 2 convolutional layers (3→32→64 channels)
- **Usage**: Simulates a smaller student model that will learn from the teacher

## Individual Tests

### `test_calculate_gradient_norm`

**Purpose**: Tests the gradient norm calculation functionality during training.

**What it tests**:
- Verifies that gradient norms can be computed correctly
- Ensures the function returns a positive float value
- Tests the gradient calculation process with dummy input and target tensors

**Key aspects**:
- Creates dummy input (1×3×32×32) and target (1×64×32×32) tensors
- Performs forward and backward passes
- Calculates gradient norm using the `calculate_gradient_norm` function
- Validates that the result is a positive float

**Importance**: Gradient norm calculation is crucial for monitoring training stability and implementing gradient clipping in the distillation process.

### `test_freeze_layers`

**Purpose**: Tests the layer freezing functionality used in transfer learning.

**What it tests**:
- Verifies that specified layers can be frozen (parameters set to non-trainable)
- Ensures that other layers remain trainable
- Tests the selective freezing mechanism

**Key aspects**:
- Freezes the first layer of the student model
- Checks that frozen layer parameters have `requires_grad=False`
- Verifies that unfrozen layers maintain `requires_grad=True`

**Importance**: Layer freezing is essential for transfer learning, allowing the model to preserve pre-trained knowledge while fine-tuning specific layers.

### `test_head_features_decoder`

**Purpose**: Tests the decoding of detection head features into bounding boxes and class scores.

**What it tests**:
- Verifies that feature maps can be properly decoded into detection predictions
- Tests the conversion from raw feature maps to bounding box coordinates and class scores
- Ensures proper tensor shapes and dimensions

**Key aspects**:
- Creates dummy feature maps for different stride levels (8, 16, 32)
- Uses a mock detection criterion for bbox decoding
- Tests the complete decoding pipeline from features to predictions
- Validates output tensor format and shape

**Importance**: This function is critical for converting the model's internal representations into usable detection predictions during distillation.

### `test_compute_distillation_loss`

**Purpose**: Tests the core distillation loss computation that combines detection loss and knowledge distillation loss.

**What it tests**:
- Verifies that distillation loss can be computed from teacher and student predictions
- Tests the combination of multiple loss components (detection, CIoU, KL divergence)
- Ensures proper loss scaling and temperature application

**Key aspects**:
- Creates dummy teacher and student predictions with proper shapes
- Applies sigmoid activation to class predictions
- Tests loss computation with various hyperparameters
- Validates that the loss is a positive scalar tensor

**Loss components tested**:
- **Detection loss**: Standard object detection loss
- **CIoU loss**: Complete IoU loss for bounding box regression
- **KL divergence**: Knowledge distillation loss between teacher and student predictions
- **Temperature scaling**: Softens probability distributions for better knowledge transfer

**Importance**: This is the core function that enables knowledge transfer from teacher to student model, making it the most critical component of the distillation pipeline.

## Test Coverage

The distillation tests cover the essential components of the knowledge distillation pipeline:

1. **Model Management**: Loading and preparing teacher/student models
2. **Training Utilities**: Gradient monitoring and layer freezing
3. **Feature Processing**: Decoding detection head features
4. **Loss Computation**: Multi-component distillation loss calculation

## Integration with Pipeline

These tests ensure that the distillation module integrates properly with the broader wildfire detection pipeline:

- **Model Optimization**: Distillation reduces model size while maintaining performance
- **Transfer Learning**: Leverages pre-trained knowledge for better wildfire detection
- **Computational Efficiency**: Smaller models enable faster inference in production
- **Quality Assurance**: Maintains detection accuracy through knowledge transfer

## Running the Tests

To run the distillation tests:

```bash
# Run all distillation tests
pytest tests/test_distillation.py -v

# Run a specific test
pytest tests/test_distillation.py::test_compute_distillation_loss -v

# Run with coverage
pytest tests/test_distillation.py --cov=pipeline.distillation
```

## Dependencies

The tests require:
- PyTorch for neural network operations
- Ultralytics YOLO for model handling
- PyYAML for configuration loading
- Pytest for test framework

## Notes

- Tests use mock models to avoid loading large pre-trained weights during testing
- Dummy data is used to simulate real-world scenarios without requiring actual wildfire images
- The tests focus on functionality rather than performance, ensuring the distillation pipeline works correctly
- All tests are designed to run quickly and independently for efficient CI/CD integration
