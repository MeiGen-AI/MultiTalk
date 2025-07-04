#!/usr/bin/env python3
"""
MultiTalk Training Script with OmegaConf and Accelerate
"""

import os
import json
import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from PIL import Image
from einops import rearrange
from tqdm import tqdm
import wandb

from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import set_seed, ProjectConfiguration
from omegaconf import DictConfig, OmegaConf

# Import your modules
import wan
from wan.configs import WAN_CONFIGS
from wan.utils.multitalk_utils import resize_and_centercrop


@dataclass
class TrainingConfig:
    """Training configuration dataclass"""
    # Model config
    model_name: str = "multitalk-14B"
    checkpoint_dir: str = "./checkpoints"
    
    # Training config
    batch_size: int = 2
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    
    # Data config
    data_root: str = "./data"
    json_file: str = "train_data.json"
    image_size: tuple = (640, 640)
    frame_num: int = 81
    max_frames: int = 1000
    
    # Audio config
    audio_window: int = 5
    vae_scale: int = 4
    
    # Logging
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Mixed precision
    mixed_precision: str = "fp16"  # fp16, bf16, no
    
    # DeepSpeed
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None


class MultiTalkDataset(Dataset):
    """Dataset for MultiTalk training"""
    
    def __init__(self, config: TrainingConfig, split: str = "train"):
        self.config = config
        self.split = split
        
        # Load data annotations
        json_path = os.path.join(config.data_root, f"{split}_{config.json_file}")
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
        self.image_size = config.image_size
        
        logging.info(f"Loaded {len(self.data)} samples for {split}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        image_path = os.path.join(self.config.data_root, item['cond_image'])
        image = Image.open(image_path).convert('RGB')
        
        # Resize and center crop
        image_tensor = resize_and_centercrop(image, self.image_size)
        image_tensor = image_tensor / 255.0
        image_tensor = (image_tensor - 0.5) * 2  # Normalize to [-1, 1]
        
        # Load audio embeddings
        audio_embeddings = []
        for person_key, audio_path in item['cond_audio'].items():
            full_audio_path = os.path.join(self.config.data_root, audio_path)
            audio_emb = torch.load(full_audio_path, map_location='cpu')
            audio_embeddings.append(audio_emb)
        
        # Process bounding boxes if available
        bboxes = None
        if 'bbox' in item:
            bboxes = []
            for person_key, bbox in item['bbox'].items():
                bboxes.append(torch.tensor(bbox, dtype=torch.float32))
            bboxes = torch.stack(bboxes)
        
        return {
            'image': image_tensor,
            'audio_embeddings': audio_embeddings,
            'bboxes': bboxes,
            'prompt': item['prompt'],
            'num_persons': len(item['cond_audio']),
            'item_id': idx
        }


def collate_fn(batch):
    """Custom collate function for batch processing"""
    images = torch.stack([item['image'] for item in batch])
    prompts = [item['prompt'] for item in batch]
    num_persons = [item['num_persons'] for item in batch]
    item_ids = [item['item_id'] for item in batch]
    
    # Handle variable number of audio embeddings
    max_persons = max(num_persons)
    batch_size = len(batch)
    
    # Pad audio embeddings
    audio_embeddings = []
    for person_idx in range(max_persons):
        person_embeddings = []
        for batch_idx in range(batch_size):
            if person_idx < len(batch[batch_idx]['audio_embeddings']):
                person_embeddings.append(batch[batch_idx]['audio_embeddings'][person_idx])
            else:
                # Pad with zeros
                dummy_shape = batch[0]['audio_embeddings'][0].shape
                person_embeddings.append(torch.zeros(dummy_shape))
        audio_embeddings.append(torch.stack(person_embeddings))
    
    # Handle bboxes
    bboxes = None
    if batch[0]['bboxes'] is not None:
        bboxes = []
        for batch_idx in range(batch_size):
            if batch[batch_idx]['bboxes'] is not None:
                bboxes.append(batch[batch_idx]['bboxes'])
            else:
                # Create dummy bboxes
                dummy_bbox = torch.zeros((max_persons, 4))
                bboxes.append(dummy_bbox)
        bboxes = torch.stack(bboxes)
    
    return {
        'images': images,
        'audio_embeddings': audio_embeddings,
        'bboxes': bboxes,
        'prompts': prompts,
        'num_persons': torch.tensor(num_persons),
        'item_ids': item_ids
    }


class MultiTalkTrainer:
    """Trainer class for MultiTalk model"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Setup accelerator
        project_config = ProjectConfiguration(
            project_dir=config.output_dir,
            logging_dir=os.path.join(config.output_dir, "logs")
        )
        
        # Setup DeepSpeed if enabled
        deepspeed_plugin = None
        if config.use_deepspeed and config.deepspeed_config:
            deepspeed_plugin = DeepSpeedPlugin(
                deepspeed_config_file=config.deepspeed_config
            )
        
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
            project_config=project_config,
            deepspeed_plugin=deepspeed_plugin
        )
        
        # Setup logging
        self.setup_logging()
        
        # Initialize model
        self.setup_model()
        
        # Setup datasets and dataloaders
        self.setup_data()
        
        # Setup optimizer and scheduler
        self.setup_optimizer()
        
        # Prepare everything with accelerator
        self.prepare_for_training()
        
        # Setup tracking
        if self.accelerator.is_main_process:
            self.setup_tracking()
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger = logging.getLogger(__name__)
        
        if self.accelerator.is_local_main_process:
            logger.info(f"Training configuration: {self.config}")
    
    def setup_model(self):
        """Initialize the MultiTalk model"""
        cfg = WAN_CONFIGS[self.config.model_name]
        
        # Initialize pipeline
        self.model = wan.MultiTalkPipeline(
            config=cfg,
            checkpoint_dir=self.config.checkpoint_dir,
            device_id=self.accelerator.device,
            rank=self.accelerator.process_index,
        )
        
        # Set model to training mode
        self.model.model.train()
        
        # Freeze certain components if needed
        self.freeze_components()
    
    def freeze_components(self):
        """Freeze certain model components"""
        # Freeze VAE
        for param in self.model.vae.parameters():
            param.requires_grad = False
        
        # Freeze text encoder
        for param in self.model.text_encoder.model.parameters():
            param.requires_grad = False
        
        # Freeze CLIP
        for param in self.model.clip.model.parameters():
            param.requires_grad = False
        
        logging.info("Frozen VAE, text encoder, and CLIP components")
    
    def setup_data(self):
        """Setup datasets and dataloaders"""
        # Create datasets
        self.train_dataset = MultiTalkDataset(self.config, split="train")
        self.val_dataset = MultiTalkDataset(self.config, split="val")
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Filter trainable parameters
        trainable_params = [p for p in self.model.model.parameters() if p.requires_grad]
        
        self.optimizer = AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8
        )
        
        # Calculate total steps
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=self.config.learning_rate * 0.01
        )
    
    def prepare_for_training(self):
        """Prepare model, optimizer, and dataloaders with accelerator"""
        (
            self.model.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler
        ) = self.accelerator.prepare(
            self.model.model,
            self.optimizer,
            self.train_dataloader,
            self.val_dataloader,
            self.scheduler
        )
    
    def setup_tracking(self):
        """Setup experiment tracking"""
        if self.accelerator.is_main_process:
            wandb.init(
                project="multitalk-training",
                config=OmegaConf.to_container(self.config, resolve=True),
                dir=self.config.output_dir
            )
    
    def compute_loss(self, batch):
        """Compute training loss"""
        images = batch['images']
        audio_embeddings = batch['audio_embeddings']
        bboxes = batch['bboxes']
        prompts = batch['prompts']
        
        # Prepare inputs for the model
        device = self.accelerator.device
        images = images.to(device)
        
        # Process audio embeddings
        processed_audio = []
        for person_audio in audio_embeddings:
            processed_audio.append(person_audio.to(device))
        
        # Create reference masks from bboxes if available
        ref_target_masks = None
        if bboxes is not None:
            ref_target_masks = self.create_masks_from_bboxes(bboxes, images.shape[-2:])
        
        # Encode text
        context = self.model.text_encoder(prompts, device)
        
        # Encode images with CLIP
        clip_context = self.model.clip.visual([images[:, :, None, :, :]])
        
        # Add noise to images (diffusion training)
        noise = torch.randn_like(images)
        timesteps = torch.randint(
            0, self.model.num_train_timesteps, 
            (images.shape[0],), 
            device=device, 
            dtype=torch.long
        )
        
        # Add noise using the scheduler
        noisy_images = self.model.add_noise(images, noise, timesteps)
        
        # Prepare model inputs
        model_inputs = {
            'x': [noisy_images[i] for i in range(noisy_images.shape[0])],
            't': timesteps,
            'context': context,
            'clip_fea': clip_context,
            'audio': torch.stack(processed_audio),
            'ref_target_masks': ref_target_masks,
            'seq_len': 75600  # This should match your model's sequence length
        }
        
        # Forward pass
        model_output = self.model.model(**model_inputs)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(model_output[0], noise, reduction="mean")
        
        return loss
    
    def create_masks_from_bboxes(self, bboxes, image_size):
        """Create reference masks from bounding boxes"""
        batch_size, num_persons, _ = bboxes.shape
        h, w = image_size
        
        masks = torch.zeros((num_persons, h, w), device=bboxes.device)
        
        for person_idx in range(num_persons):
            for batch_idx in range(batch_size):
                bbox = bboxes[batch_idx, person_idx]
                x1, y1, x2, y2 = bbox
                
                # Convert to integer coordinates
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    masks[person_idx, y1:y2, x1:x2] = 1.0
        
        return masks
    
    def train_step(self, batch):
        """Single training step"""
        with self.accelerator.accumulate(self.model.model):
            loss = self.compute_loss(batch)
            
            self.accelerator.backward(loss)
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.model.model.parameters(), 
                    self.config.max_grad_norm
                )
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.detach()
    
    def validation_step(self, batch):
        """Single validation step"""
        with torch.no_grad():
            loss = self.compute_loss(batch)
        return loss.detach()
    
    def save_checkpoint(self, epoch, step):
        """Save model checkpoint"""
        if self.accelerator.is_main_process:
            checkpoint_dir = os.path.join(
                self.config.output_dir, 
                f"checkpoint-epoch-{epoch}-step-{step}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Save model state
            self.accelerator.save_state(checkpoint_dir)
            
            # Save training config
            config_path = os.path.join(checkpoint_dir, "training_config.yaml")
            OmegaConf.save(self.config, config_path)
            
            logging.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def train(self):
        """Main training loop"""
        total_steps = 0
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for step, batch in enumerate(progress_bar):
                loss = self.train_step(batch)
                train_loss += loss.item()
                total_steps += 1
                
                # Logging
                if total_steps % self.config.logging_steps == 0:
                    avg_loss = train_loss / (step + 1)
                    lr = self.scheduler.get_last_lr()[0]
                    
                    progress_bar.set_postfix({
                        'loss': f"{avg_loss:.4f}",
                        'lr': f"{lr:.2e}"
                    })
                    
                    if self.accelerator.is_main_process:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/learning_rate': lr,
                            'train/step': total_steps,
                            'train/epoch': epoch
                        })
                
                # Validation
                if total_steps % self.config.eval_steps == 0:
                    self.validate(epoch, total_steps)
                
                # Save checkpoint
                if total_steps % self.config.save_steps == 0:
                    self.save_checkpoint(epoch, total_steps)
            
            # End of epoch validation
            self.validate(epoch, total_steps)
            
            # Save epoch checkpoint
            self.save_checkpoint(epoch, total_steps)
    
    def validate(self, epoch, step):
        """Run validation"""
        self.model.model.eval()
        val_loss = 0.0
        num_val_steps = 0
        
        for batch in self.val_dataloader:
            loss = self.validation_step(batch)
            val_loss += loss.item()
            num_val_steps += 1
        
        avg_val_loss = val_loss / num_val_steps
        
        if self.accelerator.is_main_process:
            wandb.log({
                'val/loss': avg_val_loss,
                'val/step': step,
                'val/epoch': epoch
            })
            
            logging.info(f"Validation loss: {avg_val_loss:.4f}")
        
        self.model.model.train()


def main():
    """Main training function"""
    # Load configuration
    config_path = "configs/training_config.yaml"
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        config = OmegaConf.merge(TrainingConfig(), config)
    else:
        config = TrainingConfig()
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = MultiTalkTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
