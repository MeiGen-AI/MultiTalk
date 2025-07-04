#!/usr/bin/env python3
"""
MultiTalk Training Utilities
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image
import librosa
import soundfile as sf
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model
from einops import rearrange
import pyloudnorm as pyln


class AudioProcessor:
    """Audio processing utilities for MultiTalk training"""
    
    def __init__(self, wav2vec_dir: str, device: str = 'cpu'):
        self.device = device
        self.sample_rate = 16000
        
        # Initialize Wav2Vec2 model and feature extractor
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_dir, local_files_only=True
        ).to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_dir, local_files_only=True
        )
    
    def loudness_norm(self, audio_array: np.ndarray, sr: int = 16000, lufs: float = -23) -> np.ndarray:
        """Normalize audio loudness"""
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        human_speech_array, sr = librosa.load(audio_path, sr=self.sample_rate)
        human_speech_array = self.loudness_norm(human_speech_array, sr)
        return human_speech_array
    
    def get_audio_embedding(self, speech_array: np.ndarray, video_length_frames: int = 81) -> torch.Tensor:
        """Extract audio embeddings using Wav2Vec2"""
        # Calculate video length in seconds (assuming 25 fps)
        video_length = video_length_frames / 25.0
        
        # Extract features
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=self.sample_rate).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=self.device)
        audio_feature = audio_feature.unsqueeze(0)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature, 
                seq_len=int(video_length), 
                output_hidden_states=True
            )
        
        if len(embeddings) == 0:
            raise ValueError("Failed to extract audio embedding")
        
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        
        return audio_emb.cpu().detach()


class DatasetPreprocessor:
    """Preprocessor for creating training datasets"""
    
    def __init__(self, audio_processor: AudioProcessor, output_dir: str):
        self.audio_processor = audio_processor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.audio_emb_dir = self.output_dir / "audio_embeddings"
        self.audio_emb_dir.mkdir(exist_ok=True)
    
    def process_audio_files(self, audio_paths: List[str], output_names: List[str]) -> List[str]:
        """Process audio files and save embeddings"""
        embedding_paths = []
        
        for audio_path, output_name in zip(audio_paths, output_names):
            # Load and process audio
            speech_array = self.audio_processor.load_audio(audio_path)
            
            # Extract embeddings
            audio_embedding = self.audio_processor.get_audio_embedding(speech_array)
            
            # Save embedding
            emb_path = self.audio_emb_dir / f"{output_name}.pt"
            torch.save(audio_embedding, emb_path)
            
            embedding_paths.append(str(emb_path.relative_to(self.output_dir)))
        
        return embedding_paths
    
    def create_training_data(self, raw_data: List[Dict], output_json: str):
        """Create training data from raw annotations"""
        processed_data = []
        
        for idx, item in enumerate(raw_data):
            try:
                # Process audio files
                audio_paths = []
                audio_names = []
                
                for person_key, audio_path in item['cond_audio'].items():
                    audio_paths.append(audio_path)
                    audio_names.append(f"item_{idx:06d}_{person_key}")
                
                # Extract audio embeddings
                embedding_paths = self.process_audio_files(audio_paths, audio_names)
                
                # Create processed item
                processed_item = {
                    "prompt": item["prompt"],
                    "cond_image": item["cond_image"],
                    "audio_type": item.get("audio_type", "add"),
                    "cond_audio": {},
                    "bbox": item.get("bbox", {})
                }
                
                # Update audio paths to point to embeddings
                for i, (person_key, _) in enumerate(item['cond_audio'].items()):
                    processed_item["cond_audio"][person_key] = embedding_paths[i]
                
                processed_data.append(processed_item)
                
            except Exception as e:
                print(f"Error processing item {idx}: {e}")
                continue
        
        # Save processed data
        output_path = self.output_dir / output_json
        with open(output_path, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Processed {len(processed_data)} items and saved to {output_path}")
        return str(output_path)


class BoundingBoxProcessor:
    """Utilities for processing bounding boxes"""
    
    @staticmethod
    def normalize_bbox(bbox: List[int], image_size: Tuple[int, int]) -> List[float]:
        """Normalize bounding box coordinates to [0, 1]"""
        x1, y1, x2, y2 = bbox
        h, w = image_size
        return [x1/w, y1/h, x2/w, y2/h]
    
    @staticmethod
    def denormalize_bbox(bbox: List[float], image_size: Tuple[int, int]) -> List[int]:
        """Denormalize bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        h, w = image_size
        return [int(x1*w), int(y1*h), int(x2*w), int(y2*h)]
    
    @staticmethod
    def validate_bbox(bbox: List[int], image_size: Tuple[int, int]) -> bool:
        """Validate bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        h, w = image_size
        
        # Check if coordinates are within image bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            return False
        
        # Check if bbox has positive area
        if x2 <= x1 or y2 <= y1:
            return False
        
        return True


class TrainingDataValidator:
    """Validator for training data"""
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
    
    def validate_item(self, item: Dict) -> Tuple[bool, List[str]]:
        """Validate a single data item"""
        errors = []
        
        # Check required fields
        required_fields = ['prompt', 'cond_image', 'cond_audio']
        for field in required_fields:
            if field not in item:
                errors.append(f"Missing required field: {field}")
        
        # Check image exists
        if 'cond_image' in item:
            image_path = self.data_root / item['cond_image']
            if not image_path.exists():
                errors.append(f"Image file not found: {image_path}")
        
        # Check audio embeddings exist
        if 'cond_audio' in item:
            for person_key, audio_path in item['cond_audio'].items():
                full_audio_path = self.data_root / audio_path
                if not full_audio_path.exists():
                    errors.append(f"Audio embedding not found: {full_audio_path}")
        
        # Validate bounding boxes if present
        if 'bbox' in item and 'cond_image' in item:
            try:
                # Load image to get dimensions
                image_path = self.data_root / item['cond_image']
                if image_path.exists():
                    with Image.open(image_path) as img:
                        image_size = img.size  # (width, height)
                    
                    for person_key, bbox in item['bbox'].items():
                        if not BoundingBoxProcessor.validate_bbox(bbox, image_size[::-1]):  # (height, width)
                            errors.append(f"Invalid bbox for {person_key}: {bbox}")
            except Exception as e:
                errors.append(f"Error validating bboxes: {e}")
        
        # Check consistency between audio and bbox
        if 'cond_audio' in item and 'bbox' in item:
            audio_persons = set(item['cond_audio'].keys())
            bbox_persons = set(item['bbox'].keys())
            if audio_persons != bbox_persons:
                errors.append(f"Mismatch between audio persons {audio_persons} and bbox persons {bbox_persons}")
        
        return len(errors) == 0, errors
    
    def validate_dataset(self, json_path: str) -> Tuple[int, int, List[str]]:
        """Validate entire dataset"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        valid_count = 0
        all_errors = []
        
        for idx, item in enumerate(data):
            is_valid, errors = self.validate_item(item)
            if is_valid:
                valid_count += 1
            else:
                all_errors.extend([f"Item {idx}: {error}" for error in errors])
        
        return valid_count, len(data), all_errors


def create_sample_data():
    """Create sample training data structure"""
    sample_data = [
        {
            "prompt": "A man and a woman having a friendly conversation in a living room",
            "cond_image": "images/conversation_001.jpg",
            "audio_type": "add",
            "cond_audio": {
                "person1": "audio/conversation_001_person1.wav",
                "person2": "audio/conversation_001_person2.wav"
            },
            "bbox": {
                "person1": [160, 120, 1280, 1080],
                "person2": [160, 1320, 1280, 2280]
            }
        },
        {
            "prompt": "Two people discussing work matters in an office setting",
            "cond_image": "images/office_meeting_002.jpg",
            "audio_type": "add",
            "cond_audio": {
                "person1": "audio/office_meeting_002_person1.wav",
                "person2": "audio/office_meeting_002_person2.wav"
            },
            "bbox": {
                "person1": [200, 150, 1400, 1200],
                "person2": [200, 1400, 1400, 2400]
            }
        }
    ]
    return sample_data


def main():
    """Example usage of the preprocessing utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess MultiTalk training data")
    parser.add_argument("--wav2vec_dir", type=str, required=True, help="Path to Wav2Vec2 model")
    parser.add_argument("--input_json", type=str, required=True, help="Input raw data JSON")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device for audio processing")
    
    args = parser.parse_args()
    
    # Initialize audio processor
    audio_processor = AudioProcessor(args.wav2vec_dir, args.device)
    
    # Initialize preprocessor
    preprocessor = DatasetPreprocessor(audio_processor, args.output_dir)
    
    # Load raw data
    with open(args.input_json, 'r') as f:
        raw_data = json.load(f)
    
    # Process and save training data
    train_split = int(0.8 * len(raw_data))
    train_data = raw_data[:train_split]
    val_data = raw_data[train_split:]
    
    train_path = preprocessor.create_training_data(train_data, "train_train_data.json")
    val_path = preprocessor.create_training_data(val_data, "val_train_data.json")
    
    # Validate datasets
    validator = TrainingDataValidator(args.output_dir)
    
    train_valid, train_total, train_errors = validator.validate_dataset(train_path)
    val_valid, val_total, val_errors = validator.validate_dataset(val_path)
    
    print(f"Training data: {train_valid}/{train_total} valid items")
    print(f"Validation data: {val_valid}/{val_total} valid items")
    
    if train_errors:
        print("Training data errors:")
        for error in train_errors[:10]:  # Show first 10 errors
            print(f"  {error}")
    
    if val_errors:
        print("Validation data errors:")
        for error in val_errors[:10]:  # Show first 10 errors
            print(f"  {error}")


if __name__ == "__main__":
    main()