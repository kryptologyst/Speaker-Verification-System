"""MFCC-based speaker verification model."""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from .base import BaseSpeakerModel


class MFCCSpeakerModel(BaseSpeakerModel):
    """MFCC-based speaker verification model using traditional features."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize MFCC speaker model.
        
        Args:
            config: Model configuration containing:
                - n_mfcc: Number of MFCC coefficients (default: 13)
                - n_fft: FFT window size (default: 2048)
                - hop_length: Hop length (default: 512)
                - n_mels: Number of mel filters (default: 80)
                - sample_rate: Audio sample rate (default: 16000)
        """
        super().__init__(config)
        
        self.n_mfcc = config.get("n_mfcc", 13)
        self.n_fft = config.get("n_fft", 2048)
        self.hop_length = config.get("hop_length", 512)
        self.n_mels = config.get("n_mels", 80)
        self.sample_rate = config.get("sample_rate", 16000)
        
        # MFCC extraction parameters
        self.mel_filters = self._create_mel_filters()
        
    def _create_mel_filters(self) -> torch.Tensor:
        """Create mel filter bank.
        
        Returns:
            Mel filter bank tensor
        """
        import librosa
        
        # Create mel filter bank using librosa
        mel_filters = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels
        )
        
        return torch.from_numpy(mel_filters).float()
        
    def extract_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from audio.
        
        Args:
            audio: Input audio tensor of shape (samples,) or (batch_size, samples)
            
        Returns:
            MFCC features tensor of shape (n_mfcc,) or (batch_size, n_mfcc)
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        batch_size = audio.shape[0]
        mfcc_features = []
        
        for i in range(batch_size):
            # Convert to numpy for librosa processing
            audio_np = audio[i].cpu().numpy()
            
            # Extract MFCC using librosa
            import librosa
            mfcc = librosa.feature.mfcc(
                y=audio_np,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            
            # Use mean of MFCC features across time
            mfcc_mean = torch.from_numpy(np.mean(mfcc, axis=1)).float()
            mfcc_features.append(mfcc_mean)
            
        mfcc_tensor = torch.stack(mfcc_features)
        
        if squeeze_output:
            mfcc_tensor = mfcc_tensor.squeeze(0)
            
        return mfcc_tensor.to(self.device)
        
    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            MFCC-based speaker embedding
        """
        return self.extract_mfcc(audio)
        
    def compute_similarity(self, embedding1: torch.Tensor, 
                          embedding2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Cosine similarity score
        """
        # Ensure embeddings are on CPU for sklearn
        emb1_np = embedding1.detach().cpu().numpy()
        emb2_np = embedding2.detach().cpu().numpy()
        
        # Reshape for sklearn
        if emb1_np.ndim == 1:
            emb1_np = emb1_np.reshape(1, -1)
        if emb2_np.ndim == 1:
            emb2_np = emb2_np.reshape(1, -1)
            
        # Compute cosine similarity
        similarity = cosine_similarity(emb1_np, emb2_np)[0][0]
        
        return torch.tensor(similarity, device=self.device)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            MFCC features
        """
        return self.extract_mfcc(audio)
