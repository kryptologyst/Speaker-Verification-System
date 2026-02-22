# Speaker Verification System

Research-focused speaker verification system implementing both traditional MFCC-based approaches and state-of-the-art neural network models.

## PRIVACY AND ETHICS DISCLAIMER

**IMPORTANT: This project is designed for RESEARCH and EDUCATIONAL purposes only.**

- **NOT FOR PRODUCTION USE**: This system is not intended for biometric identification in production environments
- **PRIVACY PRESERVING**: No raw personally identifiable information (PII) is logged or stored
- **RESEARCH DEMO**: Intended for academic research, education, and demonstration purposes
- **MISUSE PROHIBITION**: Voice cloning, impersonation, or any form of biometric misuse is strictly prohibited
- **ETHICAL USE**: Users must comply with applicable laws and ethical guidelines when using this software

## Features

- **Multiple Models**: MFCC baseline, x-vector, ECAPA-TDNN implementations
- **Comprehensive Evaluation**: EER, minDCF, DET curves, and speaker verification metrics
- **Modern Architecture**: PyTorch-based with proper type hints and documentation
- **Interactive Demo**: Streamlit/Gradio interface for easy experimentation
- **Research Ready**: Reproducible experiments with proper seeding and logging

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Speaker-Verification-System.git
cd Speaker-Verification-System

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For advanced features
pip install -e ".[advanced]"
```

### Basic Usage

```python
from speaker_verification import SpeakerVerifier

# Initialize verifier
verifier = SpeakerVerifier(model_type="ecapa_tdnn")

# Enroll a speaker
verifier.enroll_speaker("speaker_id", "path/to/enrollment.wav")

# Verify a speaker
result = verifier.verify("speaker_id", "path/to/test.wav")
print(f"Verification result: {result}")
```

### Demo

Launch the interactive demo:

```bash
streamlit run demo/app.py
```

## Dataset Schema

The system expects audio data in the following structure:

```
data/
├── wav/
│   ├── speaker_001/
│   │   ├── enrollment_001.wav
│   │   ├── enrollment_002.wav
│   │   └── test_001.wav
│   └── speaker_002/
│       ├── enrollment_001.wav
│       └── test_001.wav
└── meta.csv
```

The `meta.csv` file should contain:
- `id`: Unique identifier
- `path`: Path to audio file
- `speaker_id`: Speaker identifier
- `split`: train/val/test
- `duration`: Audio duration in seconds
- `sample_rate`: Audio sample rate

## Training and Evaluation

### Training

```bash
python scripts/train.py --config configs/ecapa_tdnn.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
```

### Generate Synthetic Dataset

```bash
python scripts/generate_synthetic_data.py --output_dir data/synthetic
```

## Models

### MFCC Baseline
Traditional MFCC feature extraction with cosine similarity scoring.

### x-vector
Deep neural network embedding model trained on speaker verification tasks.

### ECAPA-TDNN
Enhanced Context Aggregation for Time-Delay Neural Networks, state-of-the-art for speaker verification.

## Metrics

- **EER (Equal Error Rate)**: Point where false acceptance rate equals false rejection rate
- **minDCF (Minimum Detection Cost Function)**: Cost-weighted detection performance
- **DET Curves**: Detection Error Tradeoff curves
- **Top-1 Accuracy**: Speaker identification accuracy

## Configuration

Models and training can be configured using YAML files in the `configs/` directory:

```yaml
model:
  type: "ecapa_tdnn"
  embedding_dim: 192
  num_speakers: 1000

data:
  sample_rate: 16000
  duration: 2.0
  augment: true

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/

# Run tests
pytest tests/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Limitations

- **Research Demo**: Not suitable for production biometric systems
- **Dataset Dependent**: Performance varies significantly with training data quality
- **Computational Requirements**: Neural models require GPU for optimal performance
- **Privacy Considerations**: Always ensure compliance with privacy regulations

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{speaker_verification,
  title={Modern Speaker Verification System},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Speaker-Verification-System}
}
```
# Speaker-Verification-System
