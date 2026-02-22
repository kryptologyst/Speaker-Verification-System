#!/usr/bin/env python3
"""Setup script for speaker verification system."""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🎤 Speaker Verification System Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install package in development mode
    if not run_command("pip install -e .", "Installing package in development mode"):
        sys.exit(1)
    
    # Install development dependencies
    if not run_command("pip install -e \".[dev]\"", "Installing development dependencies"):
        sys.exit(1)
    
    # Install pre-commit hooks
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        print("⚠️  Pre-commit installation failed, but continuing...")
    
    # Create necessary directories
    directories = [
        "data/wav",
        "data/meta", 
        "checkpoints",
        "logs",
        "assets/audio",
        "assets/plots"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Generate synthetic data
    print("\n🔄 Generating synthetic dataset...")
    if run_command("python scripts/generate_synthetic_data.py --output_dir data/synthetic --num_speakers 5 --samples_per_speaker 10", 
                   "Generating synthetic dataset"):
        print("✅ Synthetic dataset generated")
    else:
        print("⚠️  Synthetic dataset generation failed, but continuing...")
    
    # Run tests
    print("\n🔄 Running tests...")
    if run_command("pytest tests/ -v", "Running tests"):
        print("✅ All tests passed")
    else:
        print("⚠️  Some tests failed, but continuing...")
    
    # Run example
    print("\n🔄 Running example...")
    if run_command("python example.py", "Running example script"):
        print("✅ Example completed successfully")
    else:
        print("⚠️  Example failed, but continuing...")
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Run the interactive demo:")
    print("   streamlit run demo/app.py")
    print("\n2. Train a model:")
    print("   python scripts/train.py --config configs/mfcc.yaml --data_dir data/synthetic --output_dir checkpoints")
    print("\n3. Evaluate a model:")
    print("   python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/synthetic --output_dir results")
    print("\n4. Run tests:")
    print("   pytest tests/ -v")
    print("\n5. Format code:")
    print("   black src/ tests/")
    print("   ruff check src/ tests/")


if __name__ == "__main__":
    main()
