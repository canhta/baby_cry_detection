# 🍼 Baby Cry Detection

AI-powered baby cry classification system with Python ML models and Flutter mobile app.

## ⚡ Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/canhta/baby_cry_detection.git
cd baby_cry_detection
./setup.sh

# 2. Activate environment  
source activate_env.sh

# 3. Train model (after organizing your dataset)
python cli.py train --model mobile --epochs 50

# 4. Convert for mobile
python cli.py convert --checkpoint models/checkpoints/best_mobilecnn_babycry.pth --model mobile

# 5. Run mobile app
cd mobile && flutter run
```

## � Project Structure

```
baby_cry_detection/
├── ml-core/              # Python ML package
├── mobile/               # Flutter mobile app  
├── cli.py               # Command-line interface
├── pyproject.toml       # Python package definition
└── setup.sh             # Development setup
```

## 🧠 Models

| Model | Size | Accuracy | Speed | Use Case |
|-------|------|----------|-------|----------|
| **MobileCNN** | 2MB | 92-95% | 10-50ms | **Production** ✅ |
| CNN | 10MB | 95% | 50-100ms | Server |
| LSTM | 15MB | 87% | 100-200ms | Research |
| CRNN | 20MB | 90-95% | 200-300ms | High accuracy |

## 📊 Supported Datasets

- **ICSD** - 10,000+ samples (Recommended)
- **Donate A Cry** - 457 samples

## 🛠️ Development

```bash
# Setup development environment
./setup.sh

# Activate environment
source activate_env.sh

# Train model
python train.py --model mobile --augment --epochs 100

# Convert to TensorFlow Lite
python convert_model.py --checkpoint [checkpoint_path] --model mobile --formats tflite

# Mobile development
cd mobile && flutter run
```

## � Mobile Features

- Real-time cry detection
- 5 cry types: Hunger, Pain, Discomfort, Tired, Normal
- Offline operation
- Confidence scoring
- Audio file analysis

## 🔧 Requirements

- Python 3.8+
- Flutter 3.0+ (for mobile)
- 4GB+ RAM
- Audio dataset

## 📚 Documentation

The `setup.sh` script handles:
- Virtual environment creation
- Dependency installation
- Dataset downloading
- Directory structure setup
- Flutter configuration

## ⭐ Key Features

✅ **Production Ready** - Optimized mobile models  
✅ **Easy Setup** - One-command installation  
✅ **Multiple Models** - Choose based on your needs  
✅ **Mobile App** - Complete Flutter implementation  
✅ **CLI Tools** - Train and convert models easily  

---

**Need help?** Check the setup script output or create an issue.