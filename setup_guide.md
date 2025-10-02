# 🍼 Baby Cry Detection - Complete Setup Guide

## 📋 Prerequisites

```bash
# Install required packages
pip install torch torchaudio librosa numpy pandas scikit-learn matplotlib
```

## 📦 Step 1: Download Datasets

### Option A: ICSD Dataset (Recommended - Most Recent)
```bash
# Clone the ICSD repository
git clone https://github.com/QingyuLiu0521/ICSD.git
cd ICSD

# Download the dataset files (follow repo instructions)
# The dataset includes:
# - Weakly labeled data: 1,888 infant cry clips
# - Strongly labeled data: 424 clips with timestamps
# - Synthetic data: 8,000 training clips
```

### Option B: Donate A Cry Corpus
```bash
git clone https://github.com/gveres/donateacry-corpus.git

# Dataset structure:
# - 457 audio files (7 seconds each)
# - Categories: hunger, burping, belly_pain, discomfort, tired
```

### Option C: Baby Chillanto Database
- Visit: http://www.inaoe.edu.mx/
- Contact: INAOE, Mexico
- 2,268 one-second clips
- Categories: asphyxia, deaf, hunger, normal, pain

### Option D: ESC-50
```bash
# Download from GitHub
git clone https://github.com/karolpiczak/ESC-50.git

# Contains 40 baby cry samples in environmental sounds
```

## 🚀 Step 2: Organize Your Data

Create this folder structure:
```
baby_cry_project/
├── data/
│   ├── train/
│   │   ├── hunger/
│   │   ├── pain/
│   │   ├── discomfort/
│   │   ├── tired/
│   │   └── normal/
│   ├── val/
│   └── test/
├── models/
└── results/
```

## 💻 Step 3: Quick Start Training Script

```python
import os
import glob
from pathlib import Path
from torch.utils.data import DataLoader
from baby_cry_detection import BabyCryDataset, CRNN_BabyCry, train_model

# 1. Prepare file paths and labels
def load_dataset(data_dir):
    audio_paths = []
    labels = []
    
    classes = ['hunger', 'pain', 'discomfort', 'tired', 'normal']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            files = glob.glob(os.path.join(class_dir, '*.wav'))
            audio_paths.extend(files)
            labels.extend([class_to_idx[class_name]] * len(files))
    
    return audio_paths, labels, classes

# 2. Load training and validation data
train_paths, train_labels, classes = load_dataset('data/train')
val_paths, val_labels, _ = load_dataset('data/val')

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")
print(f"Classes: {classes}")

# 3. Create datasets and dataloaders
train_dataset = BabyCryDataset(train_paths, train_labels)
val_dataset = BabyCryDataset(val_paths, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Initialize model
model = CRNN_BabyCry(n_classes=len(classes))

# 5. Train
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

trained_model = train_model(
    model, 
    train_loader, 
    val_loader, 
    epochs=50, 
    lr=0.001, 
    device=device
)

print("Training complete! Model saved as 'best_baby_cry_model.pth'")
```

## 🎯 Step 4: Make Predictions

```python
import torch
from baby_cry_detection import CRNN_BabyCry, predict_baby_cry

# Load trained model
model = CRNN_BabyCry(n_classes=5)
model.load_state_dict(torch.load('best_baby_cry_model.pth'))
model.eval()

# Predict on new audio
audio_file = 'test_baby_cry.wav'
predicted_class, confidence = predict_baby_cry(model, audio_file)

classes = ['hunger', 'pain', 'discomfort', 'tired', 'normal']
print(f"Predicted: {classes[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

## 📊 Step 5: Evaluate Model Performance

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.squeeze().numpy())
    
    # Print classification report
    print(classification_report(all_labels, all_preds, 
                                target_names=classes))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    return all_preds, all_labels

# Run evaluation
test_paths, test_labels, _ = load_dataset('data/test')
test_dataset = BabyCryDataset(test_paths, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32)

preds, labels = evaluate_model(model, test_loader)
```

## 🎨 Step 6: Real-Time Detection (Bonus)

```python
import sounddevice as sd
import numpy as np
import librosa

def real_time_cry_detection(model, duration=5, sr=16000):
    """
    Record audio and detect baby cry in real-time
    """
    print("Recording... (speak or play baby cry)")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    
    audio = audio.flatten()
    
    # Extract MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    
    # Predict
    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0)
    with torch.no_grad():
        output = model(mfcc_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()
        conf = probs[0][pred].item()
    
    classes = ['hunger', 'pain', 'discomfort', 'tired', 'normal']
    print(f"\nDetected: {classes[pred]} ({conf:.2%} confidence)")
    
    return pred, conf

# Use it
real_time_cry_detection(model)
```

## 🔧 Troubleshooting

### Common Issues:

1. **Out of Memory Error**
   - Reduce batch size: `batch_size=16` or `batch_size=8`
   - Use CPU instead: `device='cpu'`

2. **Low Accuracy**
   - Increase training data (data augmentation)
   - Train for more epochs
   - Use pre-trained models (transfer learning)

3. **Audio Loading Errors**
   - Ensure audio files are in WAV format
   - Convert using: `ffmpeg -i input.mp3 output.wav`

## 📈 Performance Tips

### 1. Data Augmentation
```python
# Add to your preprocessing
def augment_audio(y, sr):
    # Time stretching
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)
    
    # Pitch shifting
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    
    # Add noise
    noise = np.random.randn(len(y))
    y_noise = y + 0.005 * noise
    
    return [y, y_stretch, y_pitch, y_noise]
```

### 2. Use Pre-trained Models
```python
# Use BEATs pre-trained features (best performance)
# Download from: https://github.com/microsoft/unilm/tree/master/beats
```

### 3. Ensemble Models
```python
# Combine multiple models for better accuracy
models = [CNN_BabyCry(), LSTM_BabyCry(), CRNN_BabyCry()]
# Average predictions
```

## 📊 Expected Results

| Model | Dataset | Accuracy | F1-Score |
|-------|---------|----------|----------|
| CNN | ICSD | ~85% | 0.84 |
| LSTM | ICSD | ~87% | 0.86 |
| CRNN | ICSD | ~90% | 0.89 |
| CRNN+BEATs | ICSD | ~95% | 0.94 |

## 🌟 Next Steps

1. **Deploy as Web App**: Use Flask/FastAPI
2. **Mobile App**: Convert to TensorFlow Lite
3. **IoT Device**: Deploy on Raspberry Pi
4. **Cloud Service**: AWS Lambda or Google Cloud Functions

## 📚 Additional Resources

- Paper: "ICSD: An Open-source Dataset" (arXiv:2408.10561)
- Tutorial: Librosa documentation (https://librosa.org)
- Community: Join baby cry detection forums
- Competitions: Participate in DCASE challenges

## 🤝 Need Help?

Common questions:
- Model not converging? → Lower learning rate
- Overfitting? → Add more dropout, regularization
- Dataset too small? → Use transfer learning + data augmentation

Good luck with your baby cry detection project! 🚀
