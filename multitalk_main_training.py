#!/usr/bin/env python3
"""
MultiTalk Main Training Script
Usage: python train_multitalk.py --config-path configs --config-name training_config
"""

import os
import sys
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from accelerate.utils import set_seed

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import training modules
from multitalk_training import MultiTalkTrainer, TrainingConfig


def setup_environment(cfg: DictConfig):
    """Setup training environment"""
    # Create necessary directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.data_root, exist_ok=True)
    
    # Set random seed
    set_seed(42)
    
    # Setup CUDA if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info(f"CUDA available with {torch.cuda.device_count()} GPUs")
    else:
        logging.info("CUDA not available, using CPU")
    
    # Environment variables for optimization
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    
    # Wandb setup
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = "multitalk-training"


def validate_config(cfg: DictConfig):
    """Validate training configuration"""
    required_paths = [
        cfg.checkpoint_dir,
        cfg.data_root,
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required path does not exist: {path}")
    
    # Check data files
    train_json = os.path.join(cfg.data_root, f"train_{cfg.json_file}")
    val_json = os.path.join(cfg.data_root, f"val_{cfg.json_file}")
    
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Training data file not found: {train_json}")
    
    if not os.path.exists(val_json):
        raise FileNotFoundError(f"Validation data file not found: {val_json}")
    
    # Validate model checkpoint
    model_config_path = os.path.join(cfg.checkpoint_dir, "config.json")
    if not os.path.exists(model_config_path):
        logging.warning(f"Model config not found at {model_config_path}")
    
    logging.info("Configuration validation passed")


@hydra.main(version_base=None, config_path="configs", config_name="training_config")
def main(cfg: DictConfig) -> None:
    """Main training function"""
    
    # Convert OmegaConf to TrainingConfig dataclass
    training_config = TrainingConfig(**OmegaConf.to_container(cfg, resolve=True))
    
    # Setup environment
    setup_environment(cfg)
    
    # Validate configuration
    validate_config(cfg)
    
    # Log configuration
    logging.info("="*50)
    logging.info("MultiTalk Training Configuration")
    logging.info("="*50)
    logging.info(OmegaConf.to_yaml(cfg))
    logging.info("="*50)
    
    try:
        # Initialize trainer
        logging.info("Initializing MultiTalk trainer...")
        trainer = MultiTalkTrainer(training_config)
        
        # Start training
        logging.info("Starting training...")
        trainer.train()
        
        logging.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()


# Additional utility functions for the training script

def create_sample_config():
    """Create a sample configuration file"""
    sample_config = {
        "model_name": "multitalk-14B",
        "checkpoint_dir": "./pretrained_checkpoints",
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_epochs": 100,
        "gradient_accumulation_steps": 8,
        "max_grad_norm": 1.0,
        "warmup_steps": 1000,
        "data_root": "./data",
        "json_file": "train_data.json",
        "image_size": [640, 640],
        "frame_num": 81,
        "max_frames": 1000,
        "audio_window": 5,
        "vae_scale": 4,
        "output_dir": "./outputs",
        "logging_steps": 10,
        "save_steps": 1000,
        "eval_steps": 500,
        "resume_from_checkpoint": None,
        "mixed_precision": "fp16",
        "use_deepspeed": False,
        "deepspeed_config": None
    }
    
    config_path = "configs/training_config.yaml"
    os.makedirs("configs", exist_ok=True)
    
    with open(config_path, 'w') as f:
        OmegaConf.save(OmegaConf.create(sample_config), f)
    
    print(f"Sample configuration saved to {config_path}")


def check_system_requirements():
    """Check system requirements for training"""
    import psutil
    
    print("System Requirements Check")
    print("=" * 30)
    
    # GPU check
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory = gpu_props.total_memory / (1024**3)  # GB
            print(f"GPU {i}: {gpu_props.name}, {gpu_memory:.1f} GB")
            
            if gpu_memory < 16:
                print(f"Warning: GPU {i} has less than 16GB memory")
    else:
        print("No CUDA GPUs available")
    
    # RAM check
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"System RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 32:
        print("Warning: Less than 32GB RAM available")
    
    # Disk space check
    disk_usage = psutil.disk_usage('.')
    disk_free_gb = disk_usage.free / (1024**3)
    print(f"Free disk space: {disk_free_gb:.1f} GB")
    
    if disk_free_gb < 100:
        print("Warning: Less than 100GB free disk space")
    
    print("=" * 30)


def estimate_training_time(config_path: str):
    ""