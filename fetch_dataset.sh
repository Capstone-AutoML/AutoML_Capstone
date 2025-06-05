#!/bin/bash
if [ ! -f token.txt ]; then
    echo "token.txt not found in project, using fallback from image..."
    cp /opt/token.txt token.txt
fi

GITHUB_TOKEN=$(cat token.txt)

ZIP_DATA="mock_io/data/sampled_dataset.zip"
ZIP_MODEL="mock_io/model_registry/model_weights.zip"
IMAGE_DIR="mock_io/data/sampled_dataset/images"
MODEL_DIR="mock_io/model_registry/model"

DATA_ASSET_ID=259237441
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
