"""Speaker verification dataset."""

from typing import Dict, Any, List, Optional, Union, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import random
from collections import defaultdict

from ..utils import set_seed


class SpeakerDataset(Dataset):
    """Dataset for speaker verification."""
    
    def __init__(self, 
                 meta_file: Union[str, Path],
                 audio_dir: Union[str, Path],
                 sample_rate: int = 16000,
                 duration: float = 2.0,
                 augment: bool = False,
                 split: Optional[str] = None) -> None:
        """Initialize speaker dataset.
        
        Args:
            meta_file: Path to metadata CSV file
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate
            duration: Target duration in seconds
            augment: Whether to apply augmentation
            split: Data split (train/val/test)
        """
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment
        
        # Load metadata
        self.meta = pd.read_csv(meta_file)
        
        # Filter by split if specified
        if split is not None:
            self.meta = self.meta[self.meta['split'] == split]
            
        # Group by speaker
        self.speaker_groups = defaultdict(list)
        for _, row in self.meta.iterrows():
            self.speaker_groups[row['speaker_id']].append(row)
            
        # Create speaker list
        self.speakers = list(self.speaker_groups.keys())
        
        # Create positive and negative pairs
        self.pairs = self._create_pairs()
        
    def _create_pairs(self) -> List[Dict[str, Any]]:
        """Create positive and negative speaker pairs.
        
        Returns:
            List of speaker pairs
        """
        pairs = []
        
        # Positive pairs (same speaker)
        for speaker_id, files in self.speaker_groups.items():
            if len(files) >= 2:
                for i in range(len(files)):
                    for j in range(i + 1, len(files)):
                        pairs.append({
                            'file1': files[i]['path'],
                            'file2': files[j]['path'],
                            'speaker1': speaker_id,
                            'speaker2': speaker_id,
                            'label': 1  # Same speaker
                        })
                        
        # Negative pairs (different speakers)
        speakers = list(self.speaker_groups.keys())
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                speaker1_files = self.speaker_groups[speakers[i]]
                speaker2_files = self.speaker_groups[speakers[j]]
                
                # Create balanced negative pairs
                for _ in range(min(len(speaker1_files), len(speaker2_files))):
                    file1 = random.choice(speaker1_files)
                    file2 = random.choice(speaker2_files)
                    pairs.append({
                        'file1': file1['path'],
                        'file2': file2['path'],
                        'speaker1': speakers[i],
                        'speaker2': speakers[j],
                        'label': 0  # Different speakers
                    })
                    
        return pairs
        
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of pairs
        """
        return len(self.pairs)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing audio pairs and labels
        """
        pair = self.pairs[idx]
        
        # Load audio files
        audio1 = self._load_audio(pair['file1'])
        audio2 = self._load_audio(pair['file2'])
        
        # Apply augmentation if enabled
        if self.augment:
            audio1 = self._augment_audio(audio1)
            audio2 = self._augment_audio(audio2)
            
        return {
            'audio1': audio1,
            'audio2': audio2,
            'label': torch.tensor(pair['label'], dtype=torch.float32),
            'speaker1': pair['speaker1'],
            'speaker2': pair['speaker2']
        }
        
    def _load_audio(self, file_path: str) -> torch.Tensor:
        """Load and preprocess audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Preprocessed audio tensor
        """
        full_path = self.audio_dir / file_path
        
        # Load audio
        audio, sr = librosa.load(full_path, sr=self.sample_rate)
        
        # Pad or truncate to target duration
        target_length = int(self.duration * self.sample_rate)
        if len(audio) > target_length:
            # Random crop
            start = random.randint(0, len(audio) - target_length)
            audio = audio[start:start + target_length]
        else:
            # Pad with zeros
            audio = np.pad(audio, (0, target_length - len(audio)))
            
        return torch.from_numpy(audio).float()
        
    def _augment_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio augmentation.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Augmented audio tensor
        """
        # Convert to numpy for augmentation
        audio_np = audio.numpy()
        
        # Add noise
        if random.random() < 0.3:
            noise_level = random.uniform(0.001, 0.01)
            noise = np.random.normal(0, noise_level, audio_np.shape)
            audio_np = audio_np + noise
            
        # Speed perturbation
        if random.random() < 0.3:
            speed_factor = random.uniform(0.9, 1.1)
            audio_np = librosa.effects.time_stretch(audio_np, rate=speed_factor)
            
        # Pitch shift
        if random.random() < 0.3:
            pitch_shift = random.uniform(-2, 2)
            audio_np = librosa.effects.pitch_shift(
                audio_np, sr=self.sample_rate, n_steps=pitch_shift
            )
            
        return torch.from_numpy(audio_np).float()


class DataModule:
    """Data module for speaker verification."""
    
    def __init__(self, 
                 meta_file: Union[str, Path],
                 audio_dir: Union[str, Path],
                 batch_size: int = 32,
                 num_workers: int = 4,
                 sample_rate: int = 16000,
                 duration: float = 2.0,
                 augment_train: bool = True) -> None:
        """Initialize data module.
        
        Args:
            meta_file: Path to metadata CSV file
            audio_dir: Directory containing audio files
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            sample_rate: Target sample rate
            duration: Target duration in seconds
            augment_train: Whether to augment training data
        """
        self.meta_file = meta_file
        self.audio_dir = audio_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment_train = augment_train
        
    def setup(self) -> None:
        """Setup datasets."""
        self.train_dataset = SpeakerDataset(
            self.meta_file,
            self.audio_dir,
            sample_rate=self.sample_rate,
            duration=self.duration,
            augment=self.augment_train,
            split='train'
        )
        
        self.val_dataset = SpeakerDataset(
            self.meta_file,
            self.audio_dir,
            sample_rate=self.sample_rate,
            duration=self.duration,
            augment=False,
            split='val'
        )
        
        self.test_dataset = SpeakerDataset(
            self.meta_file,
            self.audio_dir,
            sample_rate=self.sample_rate,
            duration=self.duration,
            augment=False,
            split='test'
        )
        
    def train_dataloader(self) -> DataLoader:
        """Get training data loader.
        
        Returns:
            Training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self) -> DataLoader:
        """Get validation data loader.
        
        Returns:
            Validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self) -> DataLoader:
        """Get test data loader.
        
        Returns:
            Test data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
