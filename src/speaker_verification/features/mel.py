"""Mel spectrogram feature extraction."""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torchaudio.transforms as T


class MelSpectrogramExtractor(nn.Module):
    """Mel spectrogram feature extractor."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 80,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None,
                 power: float = 2.0,
                 normalized: bool = True) -> None:
        """Initialize mel spectrogram extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of mel filters
            fmin: Minimum frequency
            fmax: Maximum frequency
            power: Power for magnitude spectrogram
            normalized: Whether to normalize mel filters
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.power = power
        self.normalized = normalized
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
            power=power,
            normalized=normalized
        )
        
        # Log mel spectrogram
        self.log_transform = T.AmplitudeToDB()
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Mel spectrogram features tensor
        """
        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        log_mel = self.log_transform(mel_spec)
        
        return log_mel
        
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features with additional processing.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Processed mel spectrogram features
        """
        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        log_mel = self.log_transform(mel_spec)
        
        # Apply mean-variance normalization
        log_mel = self._normalize_features(log_mel)
        
        return log_mel
        
    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply mean-variance normalization.
        
        Args:
            features: Input features tensor
            
        Returns:
            Normalized features tensor
        """
        # Compute mean and std across time dimension
        mean = torch.mean(features, dim=-1, keepdim=True)
        std = torch.std(features, dim=-1, keepdim=True)
        
        # Normalize
        normalized = (features - mean) / (std + 1e-8)
        
        return normalized
        
    def get_feature_dim(self) -> int:
        """Get feature dimension.
        
        Returns:
            Feature dimension
        """
        return self.n_mels
