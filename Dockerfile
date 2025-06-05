FROM continuumio/miniconda3

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# Install make, unzip, curl
RUN apt-get update && apt-get install -y make unzip curl && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only files needed to build env
COPY environment.yml .
COPY token.txt .
COPY token.txt /opt/token.txt

# Create conda env
RUN conda env create -f environment.yml && conda clean -afy

# Use conda env for all commands
SHELL ["conda", "run", "-n", "capstone_env", "/bin/bash", "-c"]
