"""MFCC feature extraction."""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
import numpy as np


class MFCCExtractor(nn.Module):
    """MFCC feature extractor."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 n_mfcc: int = 13,
                 n_fft: int = 2048,
                 hop_length: int = 512,
                 n_mels: int = 80,
                 fmin: float = 0.0,
                 fmax: Optional[float] = None) -> None:
        """Initialize MFCC extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length
            n_mels: Number of mel filters
            fmin: Minimum frequency
            fmax: Maximum frequency
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        
        # Mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=self.fmax,
            normalized=True
        )
        
        # MFCC transform
        self.mfcc_transform = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': n_fft,
                'hop_length': hop_length,
                'n_mels': n_mels,
                'f_min': fmin,
                'f_max': self.fmax
            }
        )
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            MFCC features tensor
        """
        return self.mfcc_transform(audio)
        
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features with additional processing.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Processed MFCC features
        """
        # Extract MFCC
        mfcc = self.mfcc_transform(audio)
        
        # Apply delta and delta-delta
        delta = F.compute_deltas(mfcc)
        delta_delta = F.compute_deltas(delta)
        
        # Concatenate features
        features = torch.cat([mfcc, delta, delta_delta], dim=-2)
        
        return features
        
    def get_feature_dim(self) -> int:
        """Get feature dimension.
        
        Returns:
            Feature dimension
        """
        return self.n_mfcc * 3  # MFCC + delta + delta-delta
