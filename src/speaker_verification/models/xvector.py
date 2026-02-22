"""x-vector speaker verification model."""

from typing import Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from .base import BaseSpeakerModel


class XVectorModel(BaseSpeakerModel):
    """x-vector speaker verification model."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize x-vector model.
        
        Args:
            config: Model configuration containing:
                - input_dim: Input feature dimension (default: 40)
                - hidden_dim: Hidden layer dimension (default: 512)
                - embedding_dim: Speaker embedding dimension (default: 512)
                - num_speakers: Number of speakers for training (default: 1000)
                - dropout: Dropout rate (default: 0.5)
                - sample_rate: Audio sample rate (default: 16000)
        """
        super().__init__(config)
        
        self.input_dim = config.get("input_dim", 40)
        self.hidden_dim = config.get("hidden_dim", 512)
        self.embedding_dim = config.get("embedding_dim", 512)
        self.num_speakers = config.get("num_speakers", 1000)
        self.dropout = config.get("dropout", 0.5)
        self.sample_rate = config.get("sample_rate", 16000)
        
        # Feature extraction
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=2048,
            hop_length=512,
            n_mels=self.input_dim,
            normalized=True
        )
        
        # x-vector architecture
        self.frame_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
        # Statistics pooling
        self.pooling = StatisticsPooling()
        
        # Speaker embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )
        
        # Classification head (for training)
        self.classifier = nn.Linear(self.embedding_dim, self.num_speakers)
        
    def extract_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract mel spectrogram features.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Mel spectrogram features
        """
        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-8)
        
        # Transpose to (batch, time, freq)
        return log_mel.transpose(-2, -1)
        
    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Speaker embedding tensor
        """
        # Extract features
        features = self.extract_features(audio)
        
        # Frame-level processing
        frame_output = self.frame_layer(features)
        
        # Statistics pooling
        pooled = self.pooling(frame_output)
        
        # Speaker embedding
        embedding = self.embedding_layer(pooled)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
        
    def compute_similarity(self, embedding1: torch.Tensor, 
                          embedding2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Cosine similarity score
        """
        return F.cosine_similarity(embedding1, embedding2, dim=-1)
        
    def forward(self, audio: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple]:
        """Forward pass.
        
        Args:
            audio: Input audio tensor
            labels: Speaker labels (for training)
            
        Returns:
            Embeddings or (embeddings, logits) tuple
        """
        embedding = self.extract_embedding(audio)
        
        if labels is not None:
            logits = self.classifier(embedding)
            return embedding, logits
        else:
            return embedding


class StatisticsPooling(nn.Module):
    """Statistics pooling layer for x-vector."""
    
    def __init__(self) -> None:
        """Initialize statistics pooling."""
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply statistics pooling.
        
        Args:
            x: Input tensor of shape (batch, time, features)
            
        Returns:
            Pooled tensor of shape (batch, features * 2)
        """
        # Compute mean and std
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        
        # Concatenate mean and std
        return torch.cat([mean, std], dim=-1)
