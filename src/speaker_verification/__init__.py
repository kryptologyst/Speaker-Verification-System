"""Speaker Verification System - Main Package."""

__version__ = "1.0.0"
__author__ = "Research Team"

from .models import SpeakerVerifier
from .data import SpeakerDataset, DataModule
from .features import MFCCExtractor, MelSpectrogramExtractor
from .metrics import SpeakerVerificationMetrics

__all__ = [
    "SpeakerVerifier",
    "SpeakerDataset", 
    "DataModule",
    "MFCCExtractor",
    "MelSpectrogramExtractor",
    "SpeakerVerificationMetrics",
]
