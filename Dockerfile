FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Environment
ENV PYTHONUNBUFFERED=1
ENV PATH=/opt/conda/bin:$PATH

# Install system dependencies and Miniconda
RUN apt-get update && apt-get install -y wget bzip2 curl make unzip git && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda init bash

# Install Java (needed for IMX export)
RUN apt-get update && apt-get install -y openjdk-21-jre && apt-get clean

# Set working directory
WORKDIR /app

# Copy environment and token files
COPY environment.yml .
COPY token.txt .
COPY token.txt /opt/token.txt

# Create Conda environment with torch included
RUN sed '/torch/d' environment.yml > clean_env.yml && \
    conda update -n base -c defaults conda && \
    conda env create -f clean_env.yml && \
    conda clean -afy

# Now install GPU-compatible PyTorch explicitly (2.5.1+cu124)
RUN conda run -n capstone_env pip install \
    torch==2.5.1+cu124 \
    torchvision==0.20.1+cu124 \
    --index-url https://download.pytorch.org/whl/cu124

# Clone Grounding DINO
RUN git clone https://github.com/IDEA-Research/GroundingDINO.git /tmp/GroundingDINO

# Install it using pip (after PyTorch has been installed!)
RUN conda run -n capstone_env pip install /tmp/GroundingDINO

# # ✅ Manually build GroundingDINO extension in case pip missed it
# WORKDIR /tmp/GroundingDINO
# RUN conda run -n capstone_env bash -c "\
#     python setup.py clean && \
#     rm -rf build/ dist/ *.egg-info && \
#     pip install . \
# "

# Verify CUDA
RUN conda run -n capstone_env python -c "import torch; print('✅ Torch version:', torch.__version__); print('🧠 CUDA available:', torch.cuda.is_available())"

# Use the Conda environment shell for the rest of the image
SHELL ["conda", "run", "-n", "capstone_env", "/bin/bash", "-c"]
