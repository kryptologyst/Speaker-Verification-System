"""Test files for speaker verification."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.speaker_verification.models import SpeakerVerifier, MFCCSpeakerModel, XVectorModel, ECAPATDNNModel
from src.speaker_verification.data import SpeakerDataset, DataModule
from src.speaker_verification.features import MFCCExtractor, MelSpectrogramExtractor
from src.speaker_verification.metrics import SpeakerVerificationMetrics
from src.speaker_verification.utils import set_seed, get_device


class TestSpeakerVerifier:
    """Test speaker verifier functionality."""
    
    def test_mfcc_model_creation(self):
        """Test MFCC model creation."""
        config = {
            "n_mfcc": 13,
            "sample_rate": 16000
        }
        model = MFCCSpeakerModel(config)
        assert model is not None
        assert model.n_mfcc == 13
        
    def test_xvector_model_creation(self):
        """Test x-vector model creation."""
        config = {
            "input_dim": 40,
            "hidden_dim": 512,
            "embedding_dim": 512,
            "sample_rate": 16000
        }
        model = XVectorModel(config)
        assert model is not None
        assert model.input_dim == 40
        
    def test_ecapa_tdnn_model_creation(self):
        """Test ECAPA-TDNN model creation."""
        config = {
            "input_dim": 80,
            "channels": 512,
            "embedding_dim": 192,
            "sample_rate": 16000
        }
        model = ECAPATDNNModel(config)
        assert model is not None
        assert model.input_dim == 80
        
    def test_verifier_creation(self):
        """Test verifier creation."""
        verifier = SpeakerVerifier(model_type="mfcc")
        assert verifier is not None
        assert verifier.model_type == "mfcc"
        
    def test_verifier_enrollment(self):
        """Test speaker enrollment."""
        verifier = SpeakerVerifier(model_type="mfcc")
        
        # Create dummy audio
        audio = torch.randn(16000)  # 1 second at 16kHz
        
        # Enroll speaker
        verifier.enroll_speaker("test_speaker", audio_data=audio)
        
        # Check enrollment
        assert "test_speaker" in verifier.speaker_database
        assert verifier.speaker_database["test_speaker"].shape[0] == 13  # MFCC features
        
    def test_verifier_verification(self):
        """Test speaker verification."""
        verifier = SpeakerVerifier(model_type="mfcc")
        
        # Create dummy audio
        audio = torch.randn(16000)
        
        # Enroll speaker
        verifier.enroll_speaker("test_speaker", audio_data=audio)
        
        # Verify speaker
        result = verifier.verify_speaker("test_speaker", audio_data=audio)
        
        assert "similarity" in result
        assert "verified" in result
        assert "confidence" in result
        assert result["speaker_id"] == "test_speaker"


class TestMetrics:
    """Test metrics functionality."""
    
    def test_metrics_creation(self):
        """Test metrics creation."""
        metrics = SpeakerVerificationMetrics()
        assert metrics is not None
        
    def test_metrics_update(self):
        """Test metrics update."""
        metrics = SpeakerVerificationMetrics()
        
        similarities = torch.tensor([0.8, 0.3, 0.9, 0.2])
        labels = torch.tensor([1, 0, 1, 0])
        
        metrics.update(similarities, labels)
        
        assert len(metrics.similarities) == 4
        assert len(metrics.labels) == 4
        
    def test_metrics_computation(self):
        """Test metrics computation."""
        metrics = SpeakerVerificationMetrics()
        
        # Add some test data
        similarities = torch.tensor([0.8, 0.3, 0.9, 0.2, 0.7, 0.4])
        labels = torch.tensor([1, 0, 1, 0, 1, 0])
        
        metrics.update(similarities, labels)
        
        # Test individual metrics
        eer = metrics.compute_eer()
        auc = metrics.compute_auc()
        accuracy = metrics.compute_accuracy()
        
        assert 0 <= eer <= 1
        assert 0 <= auc <= 1
        assert 0 <= accuracy <= 1
        
        # Test all metrics
        all_metrics = metrics.get_all_metrics()
        assert "eer" in all_metrics
        assert "auc" in all_metrics
        assert "accuracy" in all_metrics


class TestFeatures:
    """Test feature extraction."""
    
    def test_mfcc_extractor(self):
        """Test MFCC feature extraction."""
        extractor = MFCCExtractor(sample_rate=16000)
        
        # Create dummy audio
        audio = torch.randn(16000)
        
        # Extract features
        features = extractor(audio)
        
        assert features.shape[0] == 13  # n_mfcc
        assert features.shape[1] > 0  # time dimension
        
    def test_mel_extractor(self):
        """Test mel spectrogram extraction."""
        extractor = MelSpectrogramExtractor(sample_rate=16000)
        
        # Create dummy audio
        audio = torch.randn(16000)
        
        # Extract features
        features = extractor(audio)
        
        assert features.shape[0] == 80  # n_mels
        assert features.shape[1] > 0  # time dimension


class TestUtils:
    """Test utility functions."""
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        # This should not raise an exception
        assert True
        
    def test_device_detection(self):
        """Test device detection."""
        device = get_device()
        assert device is not None
        assert isinstance(device, torch.device)


if __name__ == "__main__":
    pytest.main([__file__])
