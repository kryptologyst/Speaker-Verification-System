"""Speaker verification interface."""

from typing import Dict, Any, Optional, Union, List, Tuple
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_curve, auc
import json

from .base import BaseSpeakerModel
from .mfcc import MFCCSpeakerModel
from .xvector import XVectorModel
from .ecapa_tdnn import ECAPATDNNModel


class SpeakerVerifier:
    """High-level speaker verification interface."""
    
    def __init__(self, model_type: str = "mfcc", config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize speaker verifier.
        
        Args:
            model_type: Type of model to use ("mfcc", "xvector", "ecapa_tdnn")
            config: Model configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}
        
        # Initialize model
        self.model = self._create_model()
        
        # Speaker database
        self.speaker_database: Dict[str, torch.Tensor] = {}
        
        # Verification threshold
        self.threshold = self.config.get("threshold", 0.5)
        
    def _create_model(self) -> BaseSpeakerModel:
        """Create model based on type.
        
        Returns:
            Initialized speaker model
        """
        if self.model_type == "mfcc":
            return MFCCSpeakerModel(self.config)
        elif self.model_type == "xvector":
            return XVectorModel(self.config)
        elif self.model_type == "ecapa_tdnn":
            return ECAPATDNNModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def enroll_speaker(self, speaker_id: str, audio_path: Union[str, Path], 
                      audio_data: Optional[torch.Tensor] = None) -> None:
        """Enroll a speaker in the database.
        
        Args:
            speaker_id: Unique speaker identifier
            audio_path: Path to enrollment audio file
            audio_data: Optional pre-loaded audio data
        """
        if audio_data is not None:
            audio = audio_data
        else:
            audio = self._load_audio(audio_path)
            
        # Extract speaker embedding
        embedding = self.model.extract_embedding(audio)
        
        # Store in database
        self.speaker_database[speaker_id] = embedding
        
    def verify_speaker(self, speaker_id: str, audio_path: Union[str, Path], 
                      audio_data: Optional[torch.Tensor] = None,
                      threshold: Optional[float] = None) -> Dict[str, Any]:
        """Verify a speaker against enrolled speaker.
        
        Args:
            speaker_id: Speaker ID to verify against
            audio_path: Path to test audio file
            audio_data: Optional pre-loaded audio data
            threshold: Verification threshold (uses default if None)
            
        Returns:
            Verification result dictionary
        """
        if speaker_id not in self.speaker_database:
            raise ValueError(f"Speaker {speaker_id} not found in database")
            
        if audio_data is not None:
            audio = audio_data
        else:
            audio = self._load_audio(audio_path)
            
        # Extract test embedding
        test_embedding = self.model.extract_embedding(audio)
        
        # Get enrolled embedding
        enrolled_embedding = self.speaker_database[speaker_id]
        
        # Compute similarity
        similarity = self.model.compute_similarity(enrolled_embedding, test_embedding)
        
        # Apply threshold
        threshold = threshold or self.threshold
        is_verified = similarity.item() > threshold
        
        return {
            "speaker_id": speaker_id,
            "similarity": similarity.item(),
            "threshold": threshold,
            "verified": is_verified,
            "confidence": abs(similarity.item() - threshold)
        }
        
    def identify_speaker(self, audio_path: Union[str, Path], 
                       audio_data: Optional[torch.Tensor] = None,
                       top_k: int = 1) -> List[Dict[str, Any]]:
        """Identify speaker from audio.
        
        Args:
            audio_path: Path to test audio file
            audio_data: Optional pre-loaded audio data
            top_k: Number of top matches to return
            
        Returns:
            List of top-k speaker matches
        """
        if not self.speaker_database:
            raise ValueError("No speakers enrolled in database")
            
        if audio_data is not None:
            audio = audio_data
        else:
            audio = self._load_audio(audio_path)
            
        # Extract test embedding
        test_embedding = self.model.extract_embedding(audio)
        
        # Compute similarities with all enrolled speakers
        similarities = []
        for speaker_id, enrolled_embedding in self.speaker_database.items():
            similarity = self.model.compute_similarity(enrolled_embedding, test_embedding)
            similarities.append({
                "speaker_id": speaker_id,
                "similarity": similarity.item()
            })
            
        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        return similarities[:top_k]
        
    def _load_audio(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio tensor
        """
        import librosa
        
        audio, sr = librosa.load(audio_path, sr=self.config.get("sample_rate", 16000))
        return torch.from_numpy(audio).float()
        
    def set_threshold(self, threshold: float) -> None:
        """Set verification threshold.
        
        Args:
            threshold: New threshold value
        """
        self.threshold = threshold
        
    def get_speaker_database(self) -> Dict[str, Any]:
        """Get speaker database information.
        
        Returns:
            Database information dictionary
        """
        return {
            "num_speakers": len(self.speaker_database),
            "speaker_ids": list(self.speaker_database.keys()),
            "model_type": self.model_type,
            "threshold": self.threshold
        }
        
    def save_database(self, path: Union[str, Path]) -> None:
        """Save speaker database to file.
        
        Args:
            path: Path to save database
        """
        database_data = {
            "speaker_database": {
                speaker_id: embedding.cpu().numpy().tolist()
                for speaker_id, embedding in self.speaker_database.items()
            },
            "model_type": self.model_type,
            "config": self.config,
            "threshold": self.threshold
        }
        
        with open(path, 'w') as f:
            json.dump(database_data, f)
            
    def load_database(self, path: Union[str, Path]) -> None:
        """Load speaker database from file.
        
        Args:
            path: Path to database file
        """
        with open(path, 'r') as f:
            database_data = json.load(f)
            
        # Load speaker embeddings
        self.speaker_database = {
            speaker_id: torch.tensor(embedding)
            for speaker_id, embedding in database_data["speaker_database"].items()
        }
        
        # Load configuration
        self.model_type = database_data["model_type"]
        self.config = database_data["config"]
        self.threshold = database_data["threshold"]
        
        # Recreate model
        self.model = self._create_model()
        
    def compute_eer(self, similarities: List[float], labels: List[bool]) -> float:
        """Compute Equal Error Rate.
        
        Args:
            similarities: List of similarity scores
            labels: List of true/false labels
            
        Returns:
            Equal Error Rate
        """
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer
