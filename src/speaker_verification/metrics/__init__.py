"""Evaluation metrics for speaker verification."""

from .verification import SpeakerVerificationMetrics
from .detection import DetectionErrorTradeoff

__all__ = [
    "SpeakerVerificationMetrics",
    "DetectionErrorTradeoff",
]
