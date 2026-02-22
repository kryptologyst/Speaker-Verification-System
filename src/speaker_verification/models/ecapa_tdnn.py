"""ECAPA-TDNN speaker verification model."""

from typing import Dict, Any, Union, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T

from .base import BaseSpeakerModel


class ECAPATDNNModel(BaseSpeakerModel):
    """ECAPA-TDNN speaker verification model."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize ECAPA-TDNN model.
        
        Args:
            config: Model configuration containing:
                - input_dim: Input feature dimension (default: 80)
                - channels: Number of channels in TDNN layers (default: 512)
                - embedding_dim: Speaker embedding dimension (default: 192)
                - num_speakers: Number of speakers for training (default: 1000)
                - dropout: Dropout rate (default: 0.5)
                - sample_rate: Audio sample rate (default: 16000)
        """
        super().__init__(config)
        
        self.input_dim = config.get("input_dim", 80)
        self.channels = config.get("channels", 512)
        self.embedding_dim = config.get("embedding_dim", 192)
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
        
        # ECAPA-TDNN layers
        self.conv1 = nn.Conv1d(self.input_dim, self.channels, kernel_size=5, dilation=1)
        self.bn1 = nn.BatchNorm1d(self.channels)
        
        self.conv2 = nn.Conv1d(self.channels, self.channels, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm1d(self.channels)
        
        self.conv3 = nn.Conv1d(self.channels, self.channels, kernel_size=3, dilation=3)
        self.bn3 = nn.BatchNorm1d(self.channels)
        
        self.conv4 = nn.Conv1d(self.channels, self.channels, kernel_size=1, dilation=1)
        self.bn4 = nn.BatchNorm1d(self.channels)
        
        # SE-Res2Block
        self.se_res2_block = SERes2Block(self.channels, self.channels)
        
        # Attentive statistics pooling
        self.attentive_pooling = AttentiveStatisticsPooling(self.channels)
        
        # Speaker embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(self.channels * 2, self.embedding_dim),
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
        
        return log_mel
        
    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Speaker embedding tensor
        """
        # Extract features
        features = self.extract_features(audio)
        
        # TDNN layers
        x = F.relu(self.bn1(self.conv1(features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        # SE-Res2Block
        x = self.se_res2_block(x)
        
        # Attentive statistics pooling
        pooled = self.attentive_pooling(x)
        
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


class SERes2Block(nn.Module):
    """Squeeze-and-Excitation Res2Block."""
    
    def __init__(self, in_channels: int, out_channels: int, scale: int = 8) -> None:
        """Initialize SE-Res2Block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            scale: Scale factor for Res2Net
        """
        super().__init__()
        
        self.scale = scale
        self.channels = out_channels // scale
        
        # Res2Net convolutions
        self.convs = nn.ModuleList([
            nn.Conv1d(self.channels, self.channels, kernel_size=3, dilation=1)
            for _ in range(scale)
        ])
        
        # Batch normalization
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(self.channels) for _ in range(scale)
        ])
        
        # Squeeze-and-Excitation
        self.se = SEBlock(out_channels)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        residual = self.residual(x)
        
        # Split channels
        chunks = torch.chunk(x, self.scale, dim=1)
        
        # Apply Res2Net convolutions
        outputs = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0:
                out = F.relu(bn(conv(chunks[i])))
            else:
                out = F.relu(bn(conv(chunks[i] + outputs[i-1])))
            outputs.append(out)
            
        # Concatenate outputs
        out = torch.cat(outputs, dim=1)
        
        # Apply SE
        out = self.se(out)
        
        # Residual connection
        return out + residual


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels: int, reduction: int = 16) -> None:
        """Initialize SE block.
        
        Args:
            channels: Number of channels
            reduction: Reduction factor
        """
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Global average pooling
        squeezed = self.squeeze(x).squeeze(-1)
        
        # Excitation
        excitation = self.excitation(squeezed).unsqueeze(-1)
        
        # Scale
        return x * excitation


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling."""
    
    def __init__(self, channels: int) -> None:
        """Initialize attentive statistics pooling.
        
        Args:
            channels: Number of input channels
        """
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attentive statistics pooling.
        
        Args:
            x: Input tensor of shape (batch, channels, time)
            
        Returns:
            Pooled tensor of shape (batch, channels * 2)
        """
        # Compute attention weights
        attention_weights = self.attention(x)
        
        # Weighted mean
        weighted_mean = torch.sum(x * attention_weights, dim=-1)
        
        # Weighted variance
        weighted_var = torch.sum(
            (x - weighted_mean.unsqueeze(-1)) ** 2 * attention_weights, 
            dim=-1
        )
        
        # Concatenate mean and variance
        return torch.cat([weighted_mean, weighted_var], dim=-1)
