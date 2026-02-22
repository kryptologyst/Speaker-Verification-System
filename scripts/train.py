#!/usr/bin/env python3
"""Training script for speaker verification models."""

import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from src.speaker_verification.models import SpeakerVerifier
from src.speaker_verification.data import DataModule
from src.speaker_verification.metrics import SpeakerVerificationMetrics
from src.speaker_verification.utils import set_seed, get_device, setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train speaker verification model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--wandb", action="store_true", help="Use wandb logging")
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    metrics = SpeakerVerificationMetrics()
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        audio1 = batch['audio1'].to(device)
        audio2 = batch['audio2'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Extract embeddings
        emb1 = model.extract_embedding(audio1)
        emb2 = model.extract_embedding(audio2)
        
        # Compute similarities
        similarities = model.compute_similarity(emb1, emb2)
        
        # Compute loss
        loss = criterion(similarities, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metrics.update(similarities, labels)
        total_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.get_all_metrics()
    
    logger.info(f"Training - Loss: {avg_loss:.4f}, EER: {epoch_metrics['eer']:.4f}")
    
    return avg_loss, epoch_metrics


def validate_epoch(model, dataloader, criterion, device, logger):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    metrics = SpeakerVerificationMetrics()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            audio1 = batch['audio1'].to(device)
            audio2 = batch['audio2'].to(device)
            labels = batch['label'].to(device)
            
            # Extract embeddings
            emb1 = model.extract_embedding(audio1)
            emb2 = model.extract_embedding(audio2)
            
            # Compute similarities
            similarities = model.compute_similarity(emb1, emb2)
            
            # Compute loss
            loss = criterion(similarities, labels)
            
            # Update metrics
            metrics.update(similarities, labels)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
    avg_loss = total_loss / len(dataloader)
    epoch_metrics = metrics.get_all_metrics()
    
    logger.info(f"Validation - Loss: {avg_loss:.4f}, EER: {epoch_metrics['eer']:.4f}")
    
    return avg_loss, epoch_metrics


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Setup wandb if enabled
    if args.wandb:
        wandb.init(
            project="speaker-verification",
            config=config,
            name=f"{config['model']['type']}_training"
        )
    
    # Setup data
    data_module = DataModule(
        meta_file=Path(args.data_dir) / "meta.csv",
        audio_dir=Path(args.data_dir) / "wav",
        **config['data']
    )
    data_module.setup()
    
    # Setup model
    model = SpeakerVerifier(
        model_type=config['model']['type'],
        config=config['model']
    ).to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_eer = float('inf')
    for epoch in range(config['training']['epochs']):
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, data_module.train_dataloader(), optimizer, criterion, device, logger
        )
        
        # Validate
        val_loss, val_metrics = validate_epoch(
            model, data_module.val_dataloader(), criterion, device, logger
        )
        
        # Log metrics
        if args.wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_eer": train_metrics['eer'],
                "val_eer": val_metrics['eer'],
                "val_auc": val_metrics['auc'],
            })
        
        # Save best model
        if val_metrics['eer'] < best_eer:
            best_eer = val_metrics['eer']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'best_model.pth')
            
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / f'checkpoint_epoch_{epoch+1}.pth')
    
    logger.info(f"Training completed. Best EER: {best_eer:.4f}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
