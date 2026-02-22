"""Detection Error Tradeoff (DET) curve implementation."""

from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


class DetectionErrorTradeoff:
    """Detection Error Tradeoff curve calculator."""
    
    def __init__(self) -> None:
        """Initialize DET calculator."""
        self.fpr: List[float] = []
        self.fnr: List[float] = []
        self.thresholds: List[float] = []
        
    def compute_det_curve(self, similarities: List[float], labels: List[bool]) -> Tuple[List[float], List[float], List[float]]:
        """Compute DET curve.
        
        Args:
            similarities: List of similarity scores
            labels: List of true/false labels
            
        Returns:
            Tuple of (fpr, fnr, thresholds)
        """
        fpr, tpr, thresholds = roc_curve(labels, similarities)
        fnr = 1 - tpr
        
        self.fpr = fpr.tolist()
        self.fnr = fnr.tolist()
        self.thresholds = thresholds.tolist()
        
        return self.fpr, self.fnr, self.thresholds
        
    def plot_det_curve(self, save_path: Optional[str] = None) -> None:
        """Plot DET curve.
        
        Args:
            save_path: Path to save plot
        """
        if not self.fpr or not self.fnr:
            raise ValueError("DET curve not computed yet. Call compute_det_curve first.")
            
        plt.figure(figsize=(8, 6))
        plt.plot(self.fpr, self.fnr, color='darkorange', lw=2, label='DET curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('False Negative Rate')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.legend(loc="upper right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def find_eer_threshold(self) -> float:
        """Find threshold at Equal Error Rate.
        
        Returns:
            Threshold at EER
        """
        if not self.fpr or not self.fnr:
            raise ValueError("DET curve not computed yet. Call compute_det_curve first.")
            
        # Find point where FPR = FNR
        diff = np.abs(np.array(self.fpr) - np.array(self.fnr))
        eer_idx = np.argmin(diff)
        
        return self.thresholds[eer_idx]
        
    def compute_eer(self) -> float:
        """Compute Equal Error Rate.
        
        Returns:
            Equal Error Rate
        """
        if not self.fpr or not self.fnr:
            raise ValueError("DET curve not computed yet. Call compute_det_curve first.")
            
        # Find point where FPR = FNR
        diff = np.abs(np.array(self.fpr) - np.array(self.fnr))
        eer_idx = np.argmin(diff)
        
        return self.fpr[eer_idx]
