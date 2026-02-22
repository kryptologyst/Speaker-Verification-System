"""Audio preprocessing utilities."""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torchaudio.transforms as T
import numpy as np


class AudioPreprocessor(nn.Module):
    """Audio preprocessing module."""
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 target_duration: float = 2.0,
                 pre_emphasis: float = 0.97,
                 normalize: bool = True) -> None:
        """Initialize audio preprocessor.
        
        Args:
            sample_rate: Target sample rate
            target_duration: Target duration in seconds
            pre_emphasis: Pre-emphasis coefficient
            normalize: Whether to normalize audio
        """
        super().__init__()
        
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.pre_emphasis = pre_emphasis
        self.normalize = normalize
        
        # Resampling transform
        self.resample = T.Resample(orig_freq=sample_rate, new_freq=sample_rate)
        
    def forward(self, audio: torch.Tensor, 
                original_sr: Optional[int] = None) -> torch.Tensor:
        """Preprocess audio.
        
        Args:
            audio: Input audio tensor
            original_sr: Original sample rate (if different from target)
            
        Returns:
            Preprocessed audio tensor
        """
        # Resample if necessary
        if original_sr is not None and original_sr != self.sample_rate:
            resample_transform = T.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
            audio = resample_transform(audio)
            
        # Apply pre-emphasis
        if self.pre_emphasis > 0:
            audio = self._apply_pre_emphasis(audio)
            
        # Pad or truncate to target duration
        audio = self._pad_or_truncate(audio)
        
        # Normalize
        if self.normalize:
            audio = self._normalize_audio(audio)
            
        return audio
        
    def _apply_pre_emphasis(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply pre-emphasis filter.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Pre-emphasized audio tensor
        """
        # Pre-emphasis: y[n] = x[n] - alpha * x[n-1]
        emphasized = torch.zeros_like(audio)
        emphasized[0] = audio[0]
        emphasized[1:] = audio[1:] - self.pre_emphasis * audio[:-1]
        
        return emphasized
        
    def _pad_or_truncate(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad or truncate audio to target duration.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Padded or truncated audio tensor
        """
        target_length = int(self.target_duration * self.sample_rate)
        
        if len(audio) > target_length:
            # Random crop
            start = torch.randint(0, len(audio) - target_length + 1, (1,)).item()
            return audio[start:start + target_length]
        else:
            # Pad with zeros
            padding = target_length - len(audio)
            return torch.nn.functional.pad(audio, (0, padding))
            
    def _normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to unit variance.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Normalized audio tensor
        """
        std = torch.std(audio)
        if std > 0:
            return audio / std
        else:
            return audio
            
    def extract_features(self, audio: torch.Tensor, 
                        feature_type: str = "raw") -> torch.Tensor:
        """Extract features from audio.
        
        Args:
            audio: Input audio tensor
            feature_type: Type of features to extract ("raw", "mfcc", "mel")
            
        Returns:
            Feature tensor
        """
        if feature_type == "raw":
            return self.forward(audio)
        elif feature_type == "mfcc":
            from .mfcc import MFCCExtractor
            mfcc_extractor = MFCCExtractor(sample_rate=self.sample_rate)
            return mfcc_extractor(audio)
        elif feature_type == "mel":
            from .mel import MelSpectrogramExtractor
            mel_extractor = MelSpectrogramExtractor(sample_rate=self.sample_rate)
            return mel_extractor(audio)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
            
    def get_feature_dim(self, feature_type: str = "raw") -> int:
        """Get feature dimension.
        
        Args:
            feature_type: Type of features
            
        Returns:
            Feature dimension
        """
        if feature_type == "raw":
            return int(self.target_duration * self.sample_rate)
        elif feature_type == "mfcc":
            return 13 * 3  # MFCC + delta + delta-delta
        elif feature_type == "mel":
            return 80  # Number of mel bins
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
