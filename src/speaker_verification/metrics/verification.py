"""Speaker verification metrics."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt


class SpeakerVerificationMetrics:
    """Metrics for speaker verification evaluation."""
    
    def __init__(self) -> None:
        """Initialize metrics calculator."""
        self.reset()
        
    def reset(self) -> None:
        """Reset all metrics."""
        self.similarities: List[float] = []
        self.labels: List[bool] = []
        self.predictions: List[bool] = []
        
    def update(self, similarities: torch.Tensor, labels: torch.Tensor, 
               threshold: float = 0.5) -> None:
        """Update metrics with new batch.
        
        Args:
            similarities: Similarity scores
            labels: True labels (1 for same speaker, 0 for different)
            threshold: Verification threshold
        """
        # Convert to numpy
        sim_np = similarities.detach().cpu().numpy()
        lab_np = labels.detach().cpu().numpy()
        
        # Store similarities and labels
        self.similarities.extend(sim_np)
        self.labels.extend(lab_np)
        
        # Compute predictions
        pred_np = (sim_np > threshold).astype(int)
        self.predictions.extend(pred_np)
        
    def compute_eer(self) -> float:
        """Compute Equal Error Rate.
        
        Returns:
            Equal Error Rate
        """
        if not self.similarities:
            return 0.0
            
        fpr, tpr, thresholds = roc_curve(self.labels, self.similarities)
        fnr = 1 - tpr
        eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
        eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
        return eer
        
    def compute_min_dcf(self, p_target: float = 0.01, 
                       c_miss: float = 1.0, c_fa: float = 1.0) -> float:
        """Compute minimum Detection Cost Function.
        
        Args:
            p_target: Prior probability of target
            c_miss: Cost of miss
            c_fa: Cost of false alarm
            
        Returns:
            Minimum DCF
        """
        if not self.similarities:
            return 0.0
            
        # Sort similarities and labels
        sorted_indices = np.argsort(self.similarities)
        sorted_similarities = np.array(self.similarities)[sorted_indices]
        sorted_labels = np.array(self.labels)[sorted_indices]
        
        # Compute DCF for each threshold
        dcf_values = []
        for threshold in sorted_similarities:
            predictions = (sorted_similarities >= threshold).astype(int)
            
            # Compute miss and false alarm rates
            miss_rate = np.sum((sorted_labels == 1) & (predictions == 0)) / np.sum(sorted_labels == 1)
            fa_rate = np.sum((sorted_labels == 0) & (predictions == 1)) / np.sum(sorted_labels == 0)
            
            # Compute DCF
            dcf = p_target * c_miss * miss_rate + (1 - p_target) * c_fa * fa_rate
            dcf_values.append(dcf)
            
        return min(dcf_values)
        
    def compute_auc(self) -> float:
        """Compute Area Under ROC Curve.
        
        Returns:
            AUC score
        """
        if not self.similarities:
            return 0.0
            
        fpr, tpr, _ = roc_curve(self.labels, self.similarities)
        return auc(fpr, tpr)
        
    def compute_accuracy(self, threshold: float = 0.5) -> float:
        """Compute accuracy.
        
        Args:
            threshold: Verification threshold
            
        Returns:
            Accuracy score
        """
        if not self.similarities:
            return 0.0
            
        predictions = (np.array(self.similarities) > threshold).astype(int)
        labels = np.array(self.labels)
        
        return np.mean(predictions == labels)
        
    def compute_precision_recall(self, threshold: float = 0.5) -> Tuple[float, float]:
        """Compute precision and recall.
        
        Args:
            threshold: Verification threshold
            
        Returns:
            Tuple of (precision, recall)
        """
        if not self.similarities:
            return 0.0, 0.0
            
        predictions = (np.array(self.similarities) > threshold).astype(int)
        labels = np.array(self.labels)
        
        # Compute precision and recall
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
        
    def compute_f1_score(self, threshold: float = 0.5) -> float:
        """Compute F1 score.
        
        Args:
            threshold: Verification threshold
            
        Returns:
            F1 score
        """
        precision, recall = self.compute_precision_recall(threshold)
        
        if precision + recall == 0:
            return 0.0
            
        return 2 * precision * recall / (precision + recall)
        
    def get_all_metrics(self, threshold: float = 0.5) -> Dict[str, float]:
        """Compute all metrics.
        
        Args:
            threshold: Verification threshold
            
        Returns:
            Dictionary of all metrics
        """
        return {
            "eer": self.compute_eer(),
            "min_dcf": self.compute_min_dcf(),
            "auc": self.compute_auc(),
            "accuracy": self.compute_accuracy(threshold),
            "precision": self.compute_precision_recall(threshold)[0],
            "recall": self.compute_precision_recall(threshold)[1],
            "f1_score": self.compute_f1_score(threshold),
        }
        
    def plot_roc_curve(self, save_path: Optional[str] = None) -> None:
        """Plot ROC curve.
        
        Args:
            save_path: Path to save plot
        """
        if not self.similarities:
            return
            
        fpr, tpr, _ = roc_curve(self.labels, self.similarities)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_det_curve(self, save_path: Optional[str] = None) -> None:
        """Plot Detection Error Tradeoff (DET) curve.
        
        Args:
            save_path: Path to save plot
        """
        if not self.similarities:
            return
            
        fpr, tpr, thresholds = roc_curve(self.labels, self.similarities)
        fnr = 1 - tpr
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, fnr, color='darkorange', lw=2, label='DET curve')
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
