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
python baby_cry.py train --model mobile --epochs 50

# 4. Convert for mobile
python baby_cry.py convert --checkpoint models/checkpoints/best_mobilecnn_babycry.pth --model mobile

# 5. Run mobile app
cd mobile && flutter run
```

## � Project Structure

```
baby_cry_detection/
├── ml_core/              # Python ML package
├── mobile/               # Flutter mobile app  
├── baby_cry.py          # Command-line interface
├── pyproject.toml       # Python package definition
└── setup.sh             # Development setup
```

## 🧠 Models

| Model | Size | Accuracy | Speed | Use Case |
|-------|------|----------|-------|----------|
| **MobileCNN** | 2MB | 92-95% | 10-50ms | **Production** ✅ |
| CNN | 10MB | 95% | 50-100ms | Server |
| CRNN | 20MB | 90-95% | 200-300ms | High accuracy |

## 📊 Supported Datasets

- **ICSD** - 10,000+ samples (Recommended)

## 🛠️ Development

```bash
# Setup development environment
./setup.sh

# Setup with demo data for immediate testing
./setup.sh --demo-data

# Clean setup (removes existing environment)
./setup.sh --clean

# Activate environment
source activate_env.sh

# Train model
python baby_cry.py train --model mobile --epochs 50

# Convert to TensorFlow Lite
python baby_cry.py convert --checkpoint models/checkpoints/best_mobilecnn_babycry.pth --model mobile

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

### 🖥️ Device Support
- **Mac M1/M2**: Automatically uses MPS (Metal Performance Shaders) acceleration
- **NVIDIA GPU**: CUDA acceleration when available
- **CPU**: Universal fallback for all systems

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