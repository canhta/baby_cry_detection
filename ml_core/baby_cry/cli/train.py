#!/usr/bin/env python3
"""
Baby Cry Detection - Main Training Script
Train models on ICSD or other datasets
"""

import sys
import os
sys.path.append('./ml_core')

import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path

from baby_cry.models.mobile_model import MobileCNN_BabyCry
from baby_cry.models.cnn_model import CNN_BabyCry
from baby_cry.models.lstm_model import LSTM_BabyCry
from baby_cry.models.crnn_model import CRNN_BabyCry
from baby_cry.data.dataset import BabyCryDataset, load_dataset
from baby_cry.training.trainer import train_model, validate_model
from baby_cry.utils.model_utils import print_model_summary, count_parameters

def main():
    parser = argparse.ArgumentParser(description='Train Baby Cry Detection Model')
    parser.add_argument('--model', type=str, default='mobile', 
                       choices=['mobile', 'cnn', 'lstm', 'crnn'],
                       help='Model architecture to train')
    parser.add_argument('--data-dir', type=str, default='./data',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--save-dir', type=str, default='./models/checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--augment', action='store_true',
                       help='Enable data augmentation')
    parser.add_argument('--test', action='store_true',
                       help='Run evaluation on test set after training')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"🚀 Baby Cry Detection Training")
    print(f"📱 Model: {args.model}")
    print(f"💾 Device: {device}")
    print(f"📊 Dataset: {args.data_dir}")
    print("=" * 50)
    
    # Load datasets
    print("📂 Loading datasets...")
    train_paths, train_labels, classes = load_dataset(os.path.join(args.data_dir, 'train'))
    val_paths, val_labels, _ = load_dataset(os.path.join(args.data_dir, 'val'))
    
    print(f"📈 Training samples: {len(train_paths)}")
    print(f"📊 Validation samples: {len(val_paths)}")
    print(f"🏷️  Classes: {classes}")
    
    if len(train_paths) == 0:
        print("❌ No training data found! Please check your dataset directory.")
        print("Expected structure:")
        print("data/")
        print("├── train/")
        print("│   ├── hunger/")
        print("│   ├── pain/")
        print("│   ├── discomfort/")
        print("│   ├── tired/")
        print("│   └── normal/")
        print("└── val/")
        print("    ├── hunger/")
        print("    └── ...")
        return
    
    # Create datasets
    train_dataset = BabyCryDataset(train_paths, train_labels, augment=args.augment)
    val_dataset = BabyCryDataset(val_paths, val_labels, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    print("🧠 Initializing model...")
    if args.model == 'mobile':
        model = MobileCNN_BabyCry(n_classes=len(classes))
    elif args.model == 'cnn':
        model = CNN_BabyCry(n_classes=len(classes))
    elif args.model == 'lstm':
        model = LSTM_BabyCry(n_classes=len(classes))
    elif args.model == 'crnn':
        model = CRNN_BabyCry(n_classes=len(classes))
    
    # Print model summary
    input_size = (40, 157)  # MFCC shape for 5 seconds audio
    print_model_summary(model, input_size)
    
    # Train model
    print("🎯 Starting training...")
    results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_path=args.save_dir
    )
    
    # Test evaluation
    if args.test and os.path.exists(os.path.join(args.data_dir, 'test')):
        print("🧪 Running test evaluation...")
        test_paths, test_labels, _ = load_dataset(os.path.join(args.data_dir, 'test'))
        
        if len(test_paths) > 0:
            test_dataset = BabyCryDataset(test_paths, test_labels, augment=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                   shuffle=False, num_workers=4)
            
            test_results = validate_model(results['model'], test_loader, device)
            print(f"📊 Final Test Accuracy: {test_results['test_acc']:.2f}%")
    
    print("✅ Training completed successfully!")
    print(f"📁 Best model saved to: {args.save_dir}")

if __name__ == "__main__":
    main()