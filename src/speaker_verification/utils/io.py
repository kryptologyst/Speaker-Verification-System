"""I/O utilities."""

import torch
from pathlib import Path
from typing import Dict, Any, Union, Optional


def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   metrics: Dict[str, float],
                   filepath: Union[str, Path],
                   additional_info: Optional[Dict[str, Any]] = None) -> None:
    """Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        filepath: Path to save checkpoint
        additional_info: Additional information to save
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
    }
    
    if additional_info:
        checkpoint.update(additional_info)
        
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: Union[str, Path],
                   model: Optional[torch.nn.Module] = None,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return checkpoint


def save_model_info(model: torch.nn.Module, 
                   filepath: Union[str, Path]) -> None:
    """Save model information.
    
    Args:
        model: Model to analyze
        filepath: Path to save info
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_name': model.__class__.__name__,
    }
    
    # Save as text file
    with open(filepath, 'w') as f:
        f.write(f"Model: {info['model_name']}\n")
        f.write(f"Total parameters: {info['total_parameters']:,}\n")
        f.write(f"Trainable parameters: {info['trainable_parameters']:,}\n")
        f.write(f"Non-trainable parameters: {total_params - trainable_params:,}\n")
