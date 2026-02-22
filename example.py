#!/usr/bin/env python3
"""Example usage of the speaker verification system."""

import torch
import numpy as np
from pathlib import Path

from src.speaker_verification.models import SpeakerVerifier
from src.speaker_verification.utils import set_seed


def main():
    """Demonstrate speaker verification system."""
    print("🎤 Speaker Verification System Demo")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize verifier with MFCC model
    print("Initializing MFCC-based speaker verifier...")
    verifier = SpeakerVerifier(model_type="mfcc")
    
    # Generate synthetic audio data for demonstration
    print("Generating synthetic audio data...")
    sample_rate = 16000
    duration = 2.0
    
    # Create audio for speaker 1
    speaker1_audio1 = torch.randn(int(sample_rate * duration))
    speaker1_audio2 = torch.randn(int(sample_rate * duration)) + 0.1  # Slightly different
    
    # Create audio for speaker 2
    speaker2_audio = torch.randn(int(sample_rate * duration)) * 0.5  # Different characteristics
    
    # Enroll speakers
    print("Enrolling speakers...")
    verifier.enroll_speaker("speaker_001", audio_data=speaker1_audio1)
    verifier.enroll_speaker("speaker_002", audio_data=speaker2_audio)
    
    # Test verification
    print("\nTesting speaker verification...")
    
    # Test 1: Same speaker (should verify)
    print("\nTest 1: Verifying speaker_001 with similar audio...")
    result1 = verifier.verify_speaker("speaker_001", audio_data=speaker1_audio2)
    print(f"Similarity: {result1['similarity']:.4f}")
    print(f"Verified: {result1['verified']}")
    print(f"Confidence: {result1['confidence']:.4f}")
    
    # Test 2: Different speaker (should not verify)
    print("\nTest 2: Verifying speaker_001 with different speaker's audio...")
    result2 = verifier.verify_speaker("speaker_001", audio_data=speaker2_audio)
    print(f"Similarity: {result2['similarity']:.4f}")
    print(f"Verified: {result2['verified']}")
    print(f"Confidence: {result2['confidence']:.4f}")
    
    # Test 3: Speaker identification
    print("\nTest 3: Speaker identification...")
    top_matches = verifier.identify_speaker(audio_data=speaker1_audio2, top_k=2)
    print("Top matches:")
    for i, match in enumerate(top_matches):
        print(f"  {i+1}. {match['speaker_id']}: {match['similarity']:.4f}")
    
    # Display database info
    print("\nSpeaker Database Information:")
    db_info = verifier.get_speaker_database()
    print(f"Number of speakers: {db_info['num_speakers']}")
    print(f"Model type: {db_info['model_type']}")
    print(f"Threshold: {db_info['threshold']}")
    print(f"Speaker IDs: {db_info['speaker_ids']}")
    
    # Test with different model types
    print("\n" + "=" * 50)
    print("Testing different model types...")
    
    # Test x-vector model
    print("\nInitializing x-vector model...")
    xvector_verifier = SpeakerVerifier(model_type="xvector")
    
    # Enroll speakers
    xvector_verifier.enroll_speaker("speaker_001", audio_data=speaker1_audio1)
    xvector_verifier.enroll_speaker("speaker_002", audio_data=speaker2_audio)
    
    # Test verification
    result_xvector = xvector_verifier.verify_speaker("speaker_001", audio_data=speaker1_audio2)
    print(f"x-vector similarity: {result_xvector['similarity']:.4f}")
    print(f"x-vector verified: {result_xvector['verified']}")
    
    # Test ECAPA-TDNN model
    print("\nInitializing ECAPA-TDNN model...")
    ecapa_verifier = SpeakerVerifier(model_type="ecapa_tdnn")
    
    # Enroll speakers
    ecapa_verifier.enroll_speaker("speaker_001", audio_data=speaker1_audio1)
    ecapa_verifier.enroll_speaker("speaker_002", audio_data=speaker2_audio)
    
    # Test verification
    result_ecapa = ecapa_verifier.verify_speaker("speaker_001", audio_data=speaker1_audio2)
    print(f"ECAPA-TDNN similarity: {result_ecapa['similarity']:.4f}")
    print(f"ECAPA-TDNN verified: {result_ecapa['verified']}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nTo run the interactive demo:")
    print("  streamlit run demo/app.py")
    print("\nTo generate synthetic data:")
    print("  python scripts/generate_synthetic_data.py --output_dir data/synthetic")
    print("\nTo train a model:")
    print("  python scripts/train.py --config configs/mfcc.yaml --data_dir data/synthetic --output_dir checkpoints")


if __name__ == "__main__":
    main()
