#!/usr/bin/env python3
"""Generate synthetic speaker verification dataset."""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm
import random


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate sine wave.
    
    Args:
        frequency: Frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Sine wave audio
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    return np.sin(2 * np.pi * frequency * t)


def generate_chirp(start_freq: float, end_freq: float, duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate chirp signal.
    
    Args:
        start_freq: Starting frequency
        end_freq: Ending frequency
        duration: Duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Chirp audio
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    freq = start_freq + (end_freq - start_freq) * t / duration
    return np.sin(2 * np.pi * freq * t)


def generate_noise(duration: float, sample_rate: int = 16000) -> np.ndarray:
    """Generate white noise.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Noise audio
    """
    return np.random.normal(0, 0.1, int(sample_rate * duration))


def generate_speaker_audio(speaker_id: int, num_samples: int, duration: float, sample_rate: int = 16000) -> list:
    """Generate audio samples for a speaker.
    
    Args:
        speaker_id: Speaker identifier
        num_samples: Number of samples to generate
        duration: Duration of each sample
        sample_rate: Sample rate
        
    Returns:
        List of audio samples
    """
    samples = []
    
    # Define speaker characteristics
    base_freq = 100 + speaker_id * 50  # Base frequency varies by speaker
    freq_range = 50  # Frequency variation range
    
    for i in range(num_samples):
        # Generate different types of audio
        if i % 3 == 0:
            # Sine wave
            freq = base_freq + random.uniform(-freq_range, freq_range)
            audio = generate_sine_wave(freq, duration, sample_rate)
        elif i % 3 == 1:
            # Chirp
            start_freq = base_freq + random.uniform(-freq_range, freq_range)
            end_freq = base_freq + random.uniform(-freq_range, freq_range)
            audio = generate_chirp(start_freq, end_freq, duration, sample_rate)
        else:
            # Noise with speaker-specific characteristics
            audio = generate_noise(duration, sample_rate)
            # Apply speaker-specific filtering
            audio = audio * (0.5 + speaker_id * 0.1)
        
        # Add some variation
        audio = audio * random.uniform(0.8, 1.2)
        
        # Add slight noise
        noise = np.random.normal(0, 0.01, len(audio))
        audio = audio + noise
        
        samples.append(audio)
    
    return samples


def create_synthetic_dataset(output_dir: str, num_speakers: int = 10, 
                           samples_per_speaker: int = 20, duration: float = 2.0,
                           sample_rate: int = 16000):
    """Create synthetic speaker verification dataset.
    
    Args:
        output_dir: Output directory
        num_speakers: Number of speakers
        samples_per_speaker: Number of samples per speaker
        duration: Duration of each sample
        sample_rate: Sample rate
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create audio directory
    audio_dir = output_path / "wav"
    audio_dir.mkdir(exist_ok=True)
    
    # Create metadata
    metadata = []
    
    # Generate speakers
    for speaker_id in tqdm(range(num_speakers), desc="Generating speakers"):
        speaker_dir = audio_dir / f"speaker_{speaker_id:03d}"
        speaker_dir.mkdir(exist_ok=True)
        
        # Generate audio samples
        samples = generate_speaker_audio(speaker_id, samples_per_speaker, duration, sample_rate)
        
        # Split samples into train/val/test
        train_samples = int(samples_per_speaker * 0.7)
        val_samples = int(samples_per_speaker * 0.15)
        test_samples = samples_per_speaker - train_samples - val_samples
        
        sample_idx = 0
        
        # Save training samples
        for i in range(train_samples):
            filename = f"train_{i:03d}.wav"
            filepath = speaker_dir / filename
            sf.write(filepath, samples[sample_idx], sample_rate)
            
            metadata.append({
                'id': f"speaker_{speaker_id:03d}_train_{i:03d}",
                'path': f"speaker_{speaker_id:03d}/{filename}",
                'speaker_id': f"speaker_{speaker_id:03d}",
                'split': 'train',
                'duration': duration,
                'sample_rate': sample_rate
            })
            sample_idx += 1
        
        # Save validation samples
        for i in range(val_samples):
            filename = f"val_{i:03d}.wav"
            filepath = speaker_dir / filename
            sf.write(filepath, samples[sample_idx], sample_rate)
            
            metadata.append({
                'id': f"speaker_{speaker_id:03d}_val_{i:03d}",
                'path': f"speaker_{speaker_id:03d}/{filename}",
                'speaker_id': f"speaker_{speaker_id:03d}",
                'split': 'val',
                'duration': duration,
                'sample_rate': sample_rate
            })
            sample_idx += 1
        
        # Save test samples
        for i in range(test_samples):
            filename = f"test_{i:03d}.wav"
            filepath = speaker_dir / filename
            sf.write(filepath, samples[sample_idx], sample_rate)
            
            metadata.append({
                'id': f"speaker_{speaker_id:03d}_test_{i:03d}",
                'path': f"speaker_{speaker_id:03d}/{filename}",
                'speaker_id': f"speaker_{speaker_id:03d}",
                'split': 'test',
                'duration': duration,
                'sample_rate': sample_rate
            })
            sample_idx += 1
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(output_path / "meta.csv", index=False)
    
    print(f"Synthetic dataset created at {output_path}")
    print(f"Number of speakers: {num_speakers}")
    print(f"Samples per speaker: {samples_per_speaker}")
    print(f"Total samples: {len(metadata)}")
    print(f"Train samples: {len(df[df['split'] == 'train'])}")
    print(f"Val samples: {len(df[df['split'] == 'val'])}")
    print(f"Test samples: {len(df[df['split'] == 'test'])}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Generate synthetic speaker verification dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_speakers", type=int, default=10, help="Number of speakers")
    parser.add_argument("--samples_per_speaker", type=int, default=20, help="Samples per speaker")
    parser.add_argument("--duration", type=float, default=2.0, help="Duration of each sample")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    create_synthetic_dataset(
        args.output_dir,
        args.num_speakers,
        args.samples_per_speaker,
        args.duration,
        args.sample_rate
    )


if __name__ == "__main__":
    main()
