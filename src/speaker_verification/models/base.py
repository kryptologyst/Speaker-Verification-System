"""Base speaker model interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import torch
import torch.nn as nn
from pathlib import Path


class BaseSpeakerModel(nn.Module, ABC):
    """Abstract base class for speaker verification models."""
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the base speaker model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                 "mps" if torch.backends.mps.is_available() else "cpu")
        
    @abstractmethod
    def extract_embedding(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding from audio.
        
        Args:
            audio: Input audio tensor of shape (batch_size, samples) or (samples,)
            
        Returns:
            Speaker embedding tensor
        """
        pass
        
    @abstractmethod
    def compute_similarity(self, embedding1: torch.Tensor, 
                          embedding2: torch.Tensor) -> torch.Tensor:
        """Compute similarity between two speaker embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
            
        Returns:
            Similarity score tensor
        """
        pass
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Model output tensor
        """
        return self.extract_embedding(audio)
        
    def save_checkpoint(self, path: Union[str, Path], 
                       additional_info: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            additional_info: Additional information to save
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
        }
        if additional_info:
            checkpoint.update(additional_info)
            
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        return checkpoint
        
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Tensor moved to model device
        """
        return tensor.to(self.device)
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary containing model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": self.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "config": self.config,
        }
