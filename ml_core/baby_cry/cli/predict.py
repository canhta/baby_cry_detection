#!/usr/bin/env python3
"""
Quick inference script for baby cry detection
Usage: python predict.py --audio audio_file.wav --model path/to/model
"""

import sys
sys.path.append('./ml_core')

import torch
import argparse
from pathlib import Path

from baby_cry.models.mobile_model import MobileCNN_BabyCry
from baby_cry.models.cnn_model import CNN_BabyCry
from baby_cry.inference.predictor import BabyCryPredictor

def main():
    parser = argparse.ArgumentParser(description='Baby Cry Detection Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--model-type', type=str, default='mobile', 
                       choices=['mobile', 'cnn'], help='Model architecture')
    
    args = parser.parse_args()
    
    if not Path(args.audio).exists():
        print(f"❌ Audio file not found: {args.audio}")
        return
    
    if not Path(args.model).exists():
        print(f"❌ Model file not found: {args.model}")
        return
    
    print(f"🍼 Baby Cry Detection - Inference")
    print(f"📄 Audio: {args.audio}")
    print(f"🧠 Model: {args.model}")
    print("=" * 50)
    
    # Load model
    if args.model_type == 'mobile':
        model_class = MobileCNN_BabyCry
    else:
        model_class = CNN_BabyCry
    
    try:
        predictor = BabyCryPredictor.from_checkpoint(args.model, model_class)
        result = predictor.predict(args.audio)
        
        if 'error' in result:
            print(f"❌ Prediction failed: {result['error']}")
            return
        
        print(f"\n🎯 Prediction Results:")
        print(f"Cry Type: {result['cry_type'].upper()}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Description: {result['description']}")
        
        print(f"\n📊 All Probabilities:")
        for cry_type, prob in result['all_probabilities'].items():
            print(f"  {cry_type}: {prob:.1%}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()