#!/bin/bash

# Step 0: Build GroundingDINO's C++/CUDA extension if not already built
cd /tmp/GroundingDINO

echo "🛠️ Building GroundingDINO extension..."
python setup.py clean > /dev/null 2>&1
rm -rf build/ dist/ *.egg-info
pip install . > /dev/null 2>&1


cd /app  # Return to project root

# --- Token and asset download section ---

if [ ! -f token.txt ]; then
    echo "token.txt not found in project, using fallback from image..."
    cp /opt/token.txt token.txt
fi

GITHUB_TOKEN=$(cat token.txt)

ZIP_DATA="mock_io/data/sampled_dataset.zip"
ZIP_MODEL="mock_io/model_registry/model_weights.zip"
IMAGE_DIR="mock_io/data/sampled_dataset/images"
MODEL_DIR="mock_io/model_registry/model"

DATA_ASSET_ID=262360637
MODEL_ASSET_ID=259250340

# Dataset
if [ ! -d "$IMAGE_DIR" ]; then
    echo "📥 Downloading dataset..."
    mkdir -p "mock_io/data/sampled_dataset"
    curl -L \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/octet-stream" \
      -o "$ZIP_DATA" \
      https://api.github.com/repos/Elshaday-Tamire/capstone-assets/releases/assets/$DATA_ASSET_ID
    unzip "$ZIP_DATA" -d "mock_io/data/sampled_dataset/"
    rm "$ZIP_DATA"
else
    echo "✅ Dataset already exists."
fi

# Model
if [ ! -d "$MODEL_DIR" ]; then
    echo "📥 Downloading model weights..."
    mkdir -p "$MODEL_DIR"
    curl -L \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/octet-stream" \
      -o "$ZIP_MODEL" \
      https://api.github.com/repos/Elshaday-Tamire/capstone-assets/releases/assets/$MODEL_ASSET_ID
    unzip "$ZIP_MODEL" -d "mock_io/model_registry/"
    rm "$ZIP_MODEL"
else
    echo "✅ Model weights already exist."
fi

# Distillation dataset
DISTILLATION_DIR="mock_io/data/distillation/distillation_dataset"
ZIP_DISTILLATION="mock_io/data/distillation_dataset.zip"
DISTILLATION_ASSET_ID=262351896

if [ ! -d "$DISTILLATION_DIR" ]; then
    echo "📥 Downloading distillation dataset..."
    mkdir -p "mock_io/data/distillation"
    curl -L \
      -H "Authorization: token $GITHUB_TOKEN" \
      -H "Accept: application/octet-stream" \
      -o "$ZIP_DISTILLATION" \
      https://api.github.com/repos/Elshaday-Tamire/capstone-assets/releases/assets/$DISTILLATION_ASSET_ID
    unzip "$ZIP_DISTILLATION" -d "mock_io/data/distillation/"
    rm "$ZIP_DISTILLATION"
else
    echo "✅ Distillation dataset already exists."
fi
