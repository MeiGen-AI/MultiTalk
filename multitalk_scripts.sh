#!/bin/bash
# scripts/preprocess_data.sh
# Preprocess raw data for MultiTalk training

set -e

# Configuration
WAV2VEC_DIR="./pretrained_models/wav2vec2"
INPUT_JSON="./raw_data/annotations.json"
OUTPUT_DIR="./data"
DEVICE="cuda:0"

echo "Starting data preprocessing..."

# Create output directory
mkdir -p $OUTPUT_DIR

# Run preprocessing
python wan/utils/multitalk_utils.py \
    --wav2vec_dir $WAV2VEC_DIR \
    --input_json $INPUT_JSON \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE

echo "Data preprocessing completed!"

---

#!/bin/bash
# scripts/train_single_gpu.sh
# Single GPU training script

set -e

# Configuration
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="multitalk-training"

# Training parameters
CONFIG_FILE="configs/training_config.yaml"
OUTPUT_DIR="./outputs/single_gpu_$(date +%Y%m%d_%H%M%S)"

echo "Starting single GPU training..."
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Copy config to output directory
cp $CONFIG_FILE $OUTPUT_DIR/

# Run training
python train_multitalk.py \
    --config-path configs \
    --config-name training_config \
    output_dir=$OUTPUT_DIR \
    mixed_precision=fp16

echo "Training completed!"

---

#!/bin/bash
# scripts/train_multi_gpu.sh
# Multi-GPU training script with accelerate

set -e

# Configuration
export WANDB_PROJECT="multitalk-training"

# Training parameters
CONFIG_FILE="configs/training_config.yaml"
OUTPUT_DIR="./outputs/multi_gpu_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=4

echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Copy config to output directory
cp $CONFIG_FILE $OUTPUT_DIR/

# Run training with accelerate
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes $NUM_GPUS \
    --main_process_port 29500 \
    train_multitalk.py \
    --config-path configs \
    --config-name training_config \
    output_dir=$OUTPUT_DIR \
    mixed_precision=fp16

echo "Multi-GPU training completed!"

---

#!/bin/bash
# scripts/train_deepspeed.sh
# DeepSpeed training script

set -e

# Configuration
export WANDB_PROJECT="multitalk-training"

# Training parameters
CONFIG_FILE="configs/training_config.yaml"
DEEPSPEED_CONFIG="configs/deepspeed_config.json"
OUTPUT_DIR="./outputs/deepspeed_$(date +%Y%m%d_%H%M%S)"
NUM_GPUS=8

echo "Starting DeepSpeed training with $NUM_GPUS GPUs..."
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p $OUTPUT_DIR

# Copy configs to output directory
cp $CONFIG_FILE $OUTPUT_DIR/
cp $DEEPSPEED_CONFIG $OUTPUT_DIR/

# Run training with DeepSpeed
accelerate launch \
    --config_file configs/accelerate_deepspeed_config.yaml \
    --num_processes $NUM_GPUS \
    --main_process_port 29500 \
    train_multitalk.py \
    --config-path configs \
    --config-name training_config \
    output_dir=$OUTPUT_DIR \
    use_deepspeed=true \
    deepspeed_config=$DEEPSPEED_CONFIG \
    mixed_precision=fp16

echo "DeepSpeed training completed!"

---

#!/bin/bash
# scripts/resume_training.sh
# Resume training from checkpoint

set -e

# Configuration
CHECKPOINT_PATH="./outputs/checkpoint-epoch-10-step-5000"
CONFIG_FILE="configs/training_config.yaml"

if [ ! -d "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint directory not found: $CHECKPOINT_PATH"
    exit 1
fi

echo "Resuming training from: $CHECKPOINT_PATH"

# Run training
accelerate launch \
    --config_file configs/accelerate_config.yaml \
    --num_processes 4 \
    --main_process_port 29500 \
    train_multitalk.py \
    --config-path configs \
    --config-name training_config \
    resume_from_checkpoint=$CHECKPOINT_PATH

echo "Training resumed and completed!"

---

# requirements.txt
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Training frameworks
accelerate>=0.25.0
transformers>=4.35.0
diffusers>=0.24.0
deepspeed>=0.12.0

# Configuration and logging
omegaconf>=2.3.0
wandb>=0.16.0
tensorboard>=2.14.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0
pyloudnorm>=0.1.1

# Image processing
pillow>=9.0.0
opencv-python>=4.8.0

# Utilities
numpy>=1.24.0
scipy>=1.10.0
einops>=0.7.0
tqdm>=4.64.0
matplotlib>=3.6.0

# Development
pytest>=7.0.0
black>=23.0.0
flake8>=6.0.0

---

# configs/accelerate_config.yaml
# Accelerate configuration for multi-GPU training

compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

---

# configs/accelerate_deepspeed_config.yaml
# Accelerate configuration for DeepSpeed training

compute_environment: LOCAL_MACHINE
deepspeed_config:
  deepspeed_config_file: configs/deepspeed_config.json
  zero3_init_flag: false
  zero3_save_16bit_model: false
distributed_type: DEEPSPEED
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

---

# Makefile
# Makefile for MultiTalk training

.PHONY: install setup preprocess train train-multi train-deepspeed resume clean

# Installation
install:
	pip install -r requirements.txt

setup:
	mkdir -p data outputs logs configs pretrained_models
	mkdir -p data/images data/audio data/audio_embeddings

# Data preprocessing
preprocess:
	bash scripts/preprocess_data.sh

# Training options
train:
	bash scripts/train_single_gpu.sh

train-multi:
	bash scripts/train_multi_gpu.sh

train-deepspeed:
	bash scripts/train_deepspeed.sh

resume:
	bash scripts/resume_training.sh

# Utilities
clean:
	rm -rf outputs/*/checkpoint-*
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

# Validation
validate-data:
	python -c "from utils.multitalk_utils import TrainingDataValidator; \
	           validator = TrainingDataValidator('data'); \
	           valid, total, errors = validator.validate_dataset('data/train_train_data.json'); \
	           print(f'Valid: {valid}/{total}'); \
	           [print(f'Error: {e}') for e in errors[:10]]"

# Model testing
test-model:
	python -c "import torch; \
	           from train_multitalk import MultiTalkTrainer, TrainingConfig; \
	           config = TrainingConfig(); \
	           trainer = MultiTalkTrainer(config); \
	           print('Model loaded successfully')"

---

# docker/Dockerfile
# Docker configuration for MultiTalk training

FROM nvidia/cuda:11.8-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set permissions
RUN chmod +x scripts/*.sh

# Default command
CMD ["bash"]

---

# docker/docker-compose.yml
# Docker Compose for MultiTalk training

version: '3.8'

services:
  multitalk-train:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_PROJECT=multitalk-training
    volumes:
      - ../data:/workspace/data
      - ../outputs:/workspace/outputs
      - ../pretrained_models:/workspace/pretrained_models
    working_dir: /workspace
    command: bash scripts/train_multi_gpu.sh
    
  multitalk-preprocess:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    volumes:
      - ../raw_data:/workspace/raw_data
      - ../data:/workspace/data
      - ../pretrained_models:/workspace/pretrained_models
    working_dir: /workspace
    command: bash scripts/preprocess_data.sh