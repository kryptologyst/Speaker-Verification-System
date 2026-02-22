#!/usr/bin/env python3
"""Evaluation script for speaker verification models."""

import argparse
import yaml
from pathlib import Path
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.speaker_verification.models import SpeakerVerifier
from src.speaker_verification.data import DataModule
from src.speaker_verification.metrics import SpeakerVerificationMetrics
from src.speaker_verification.utils import get_device, setup_logging, get_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate speaker verification model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory")
    return parser.parse_args()


def evaluate_model(model, dataloader, device, logger):
    """Evaluate model on dataset."""
    model.eval()
    metrics = SpeakerVerificationMetrics()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluation")
        for batch in pbar:
            audio1 = batch['audio1'].to(device)
            audio2 = batch['audio2'].to(device)
            labels = batch['label'].to(device)
            
            # Extract embeddings
            emb1 = model.extract_embedding(audio1)
            emb2 = model.extract_embedding(audio2)
            
            # Compute similarities
            similarities = model.compute_similarity(emb1, emb2)
            
            # Update metrics
            metrics.update(similarities, labels)
            
    return metrics


def create_leaderboard(results, output_dir):
    """Create evaluation leaderboard."""
    df = pd.DataFrame(results)
    
    # Sort by EER (lower is better)
    df = df.sort_values('eer')
    
    # Save to CSV
    df.to_csv(output_dir / 'leaderboard.csv', index=False)
    
    # Create summary
    summary = {
        'Best EER': df['eer'].min(),
        'Best AUC': df['auc'].max(),
        'Best Accuracy': df['accuracy'].max(),
        'Average EER': df['eer'].mean(),
        'Average AUC': df['auc'].mean(),
        'Average Accuracy': df['accuracy'].mean(),
    }
    
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Speaker Verification Evaluation Summary\n")
        f.write("=" * 40 + "\n")
        for metric, value in summary.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return df


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = get_logger(__name__)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Setup device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Setup model
    model = SpeakerVerifier(
        model_type=config['model']['type'],
        config=config['model']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Setup data
    data_module = DataModule(
        meta_file=Path(args.data_dir) / "meta.csv",
        audio_dir=Path(args.data_dir) / "wav",
        **config['data']
    )
    data_module.setup()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, data_module.test_dataloader(), device, logger)
    test_results = test_metrics.get_all_metrics()
    
    # Create plots
    if config['evaluation']['save_plots']:
        test_metrics.plot_roc_curve(output_dir / 'roc_curve.png')
        test_metrics.plot_det_curve(output_dir / 'det_curve.png')
    
    # Create leaderboard
    results = [{
        'model': config['model']['type'],
        'epoch': checkpoint['epoch'],
        **test_results
    }]
    
    leaderboard = create_leaderboard(results, output_dir)
    
    # Print results
    logger.info("Evaluation Results:")
    logger.info(f"EER: {test_results['eer']:.4f}")
    logger.info(f"AUC: {test_results['auc']:.4f}")
    logger.info(f"Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Min DCF: {test_results['min_dcf']:.4f}")
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
