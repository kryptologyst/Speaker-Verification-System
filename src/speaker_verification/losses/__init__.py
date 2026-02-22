"""Loss functions for speaker verification."""

from .contrastive import ContrastiveLoss
from .triplet import TripletLoss

__all__ = [
    "ContrastiveLoss",
    "TripletLoss",
]
