"""Audio augmentation utilities."""

from typing import Union, Optional
import torch
import torchaudio.transforms as T
import numpy as np
import random


class AudioAugmentation:
    """Audio augmentation class."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 noise_prob: float = 0.3,
                 speed_prob: float = 0.3,
                 pitch_prob: float = 0.3,
                 reverb_prob: float = 0.2) -> None:
        """Initialize audio augmentation.
        
        Args:
            sample_rate: Audio sample rate
            noise_prob: Probability of adding noise
            speed_prob: Probability of speed perturbation
            pitch_prob: Probability of pitch shift
            reverb_prob: Probability of adding reverb
        """
        self.sample_rate = sample_rate
        self.noise_prob = noise_prob
        self.speed_prob = speed_prob
        self.pitch_prob = pitch_prob
        self.reverb_prob = reverb_prob
        
        # Initialize transforms
        self.speed_transform = T.SpeedPerturbation(
            orig_freq=sample_rate,
            speeds=[0.9, 1.0, 1.1]
        )
        
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Augmented audio tensor
        """
        # Add noise
        if random.random() < self.noise_prob:
            audio = self._add_noise(audio)
            
        # Speed perturbation
        if random.random() < self.speed_prob:
            audio = self._speed_perturb(audio)
            
        # Pitch shift
        if random.random() < self.pitch_prob:
            audio = self._pitch_shift(audio)
            
        # Add reverb
        if random.random() < self.reverb_prob:
            audio = self._add_reverb(audio)
            
        return audio
        
    def _add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Audio with added noise
        """
        noise_level = random.uniform(0.001, 0.01)
        noise = torch.randn_like(audio) * noise_level
        return audio + noise
        
    def _speed_perturb(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply speed perturbation.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Speed perturbed audio
        """
        # Use torchaudio speed perturbation
        speed_factor = random.choice([0.9, 1.1])
        return self.speed_transform(audio.unsqueeze(0)).squeeze(0)
        
    def _pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply pitch shift.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Pitch shifted audio
        """
        # Convert to numpy for librosa
        audio_np = audio.numpy()
        
        # Apply pitch shift
        pitch_shift = random.uniform(-2, 2)
        import librosa
        shifted = librosa.effects.pitch_shift(
            audio_np, sr=self.sample_rate, n_steps=pitch_shift
        )
        
        return torch.from_numpy(shifted).float()
        
    def _add_reverb(self, audio: torch.Tensor) -> torch.Tensor:
        """Add reverb to audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Audio with reverb
        """
        # Simple reverb simulation using convolution
        reverb_length = random.randint(1000, 5000)
        reverb = torch.randn(reverb_length) * 0.1
        reverb = reverb * torch.exp(-torch.arange(reverb_length).float() / reverb_length)
        
        # Apply convolution
        padded_audio = torch.nn.functional.pad(audio, (0, reverb_length - 1))
        reverb_audio = torch.nn.functional.conv1d(
            padded_audio.unsqueeze(0).unsqueeze(0),
            reverb.unsqueeze(0).unsqueeze(0),
            padding=0
        ).squeeze()
        
        # Mix with original
        mix_ratio = random.uniform(0.1, 0.3)
        return audio * (1 - mix_ratio) + reverb_audio * mix_ratio