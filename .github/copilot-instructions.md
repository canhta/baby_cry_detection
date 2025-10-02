# Baby Cry Detection - AI Coding Agent Instructions

## Project Architecture

This is a **dual-platform AI system**: Python ML backend + Flutter mobile app for real-time baby cry classification.

**Key Components:**
- `ml_core/baby_cry/` - Core ML package (PyTorch models, training, inference)
- `mobile/` - Flutter app with TensorFlow Lite integration
- `baby_cry.py` - Main CLI entry point (routes to subcommands)
- `setup.sh` - Complete development environment setup

## Critical Workflows

### Setup & Environment
```bash
./setup.sh          # One-command setup (creates venv, installs deps, configures Flutter)
source activate_env.sh   # Always activate environment before Python work
```

### Model Development
```bash
# Train models (mobile is production-optimized)
python baby_cry.py train --model mobile --epochs 50

# Convert for mobile deployment (TensorFlow Lite)
python baby_cry.py convert --checkpoint models/checkpoints/best_mobilecnn_babycry.pth --model mobile

# Prediction
python baby_cry.py predict --audio test.wav --model models/best_model.pth
```

### Mobile Development
```bash
cd mobile && flutter run   # Run Flutter app
# Models go in mobile/assets/models/ as .tflite files
```

## Model Architecture Patterns

**Mobile-First Design Philosophy:** All models optimized for mobile deployment, especially `MobileCNN_BabyCry`:
- Uses depthwise separable convolutions (83% parameter reduction)
- Global average pooling instead of large FC layers
- Fixed 5-second audio input (80K samples at 16kHz)
- 40 MFCC features → 1D CNN → 5 classes (Hunger/Pain/Discomfort/Tired/Normal)

**Model Selection Hierarchy:**
1. `mobile` (MobileCNN) - **Production default** (2MB, 92-95% accuracy, 10-50ms)
2. `cnn` - Server deployment (10MB, 95% accuracy)
3. `crnn` - Research/high-accuracy scenarios

## Project-Specific Conventions

### File Organization
- **ML models:** `ml_core/baby_cry/models/` - Each model in separate file with consistent `__init__(n_classes=5)` interface
- **CLI commands:** `ml_core/baby_cry/cli/` - Subcommands imported dynamically in `baby_cry.py`
- **Data pipeline:** `BabyCryDataset` handles ICSD format with MFCC preprocessing
- **Mobile models:** Must be converted to `.tflite` format in `mobile/assets/models/`

### Development Patterns
- **Path handling:** Always use `sys.path.insert(0, 'ml_core')` for imports
- **Model loading:** Follow `mobile_model.py` pattern - include metadata, preprocessing, and TFLite conversion methods
- **Audio processing:** Fixed 16kHz, 5-second clips, 40 MFCC features (see `dataset.py`)
- **CLI argument pattern:** Each subcommand has its own `main()` function with argparse

### Dataset Integration
- **Expected structure:** `data/[dataset_name]/[cry_type]/audio_files.wav`
- **Supported datasets:** ICSD (recommended)
- **Label mapping:** 0=Normal, 1=Hunger, 2=Pain, 3=Discomfort, 4=Tired

## Integration Points

**Python ↔ Flutter Bridge:**
- Models converted via `convert_model.py` (PyTorch → ONNX → TensorFlow → TFLite)
- Flutter loads `.tflite` files using `tflite_flutter` plugin
- Audio preprocessing must match exactly between Python training and Flutter inference

**Key Dependencies:**
- **Python:** PyTorch, TensorFlow, librosa, torchaudio (see `pyproject.toml`)
- **Flutter:** `tflite_flutter`, `record`, `audioplayers` (see `pubspec.yaml`)

## Debugging & Testing

- **Model validation:** Use `trainer.py validate_model()` function
- **Audio pipeline:** Test MFCC extraction in `utils/audio_utils.py`
- **Mobile inference:** Check TFLite model loading in Flutter with sample audio
- **Setup issues:** `setup.sh` creates detailed logs for troubleshooting

When adding new models, ensure TFLite compatibility and mobile optimization. When modifying audio processing, maintain consistency between training and inference pipelines.