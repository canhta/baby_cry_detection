#!/usr/bin/env python3
"""
Baby Cry Detection - Model Conversion Script
Convert trained PyTorch models to mobile formats (ONNX, TensorFlow Lite)
"""

import sys
sys.path.append('./ml_core')

import torch
import argparse
from pathlib import Path
import numpy as np

from baby_cry.models.mobile_model import MobileCNN_BabyCry
from baby_cry.models.cnn_model import CNN_BabyCry
from baby_cry.models.crnn_model import CRNN_BabyCry
from baby_cry.utils.model_utils import convert_to_onnx, get_model_size_mb

def convert_to_tflite(onnx_path: str, tflite_path: str):
    """Convert ONNX model to TensorFlow Lite"""
    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        
        # Export to TensorFlow SavedModel
        tf_model_path = tflite_path.replace('.tflite', '_tf_model')
        tf_rep.export_graph(tf_model_path)
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        
        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        # Convert
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        print(f"✅ TensorFlow Lite model saved: {tflite_path}")
        print(f"📏 TFLite model size: {len(tflite_model) / 1024:.1f} KB")
        
        return True
        
    except ImportError:
        print("❌ TensorFlow or onnx-tf not installed. Install with:")
        print("pip install tensorflow onnx-tf")
        return False
    except Exception as e:
        print(f"❌ TFLite conversion failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Convert Baby Cry Detection Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch model checkpoint')
    parser.add_argument('--model', type=str, required=True,
                       choices=['mobile', 'cnn', 'crnn'],
                       help='Model architecture')
    parser.add_argument('--output-dir', type=str, default='./models/mobile',
                       help='Output directory for converted models')
    parser.add_argument('--formats', nargs='+', default=['onnx'],
                       choices=['onnx', 'tflite'],
                       help='Output formats to generate')
    
    args = parser.parse_args()
    
    print(f"🔄 Baby Cry Detection Model Conversion")
    print(f"📱 Model: {args.model}")
    print(f"📂 Checkpoint: {args.checkpoint}")
    print(f"📁 Output: {args.output_dir}")
    print(f"🎯 Formats: {args.formats}")
    print("=" * 50)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("📥 Loading model...")
    device = 'cpu'  # Use CPU for conversion
    
    # Initialize model
    if args.model == 'mobile':
        model = MobileCNN_BabyCry(n_classes=5)
    elif args.model == 'cnn':
        model = CNN_BabyCry(n_classes=5)
    elif args.model == 'crnn':
        model = CRNN_BabyCry(n_classes=5)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📏 Model size: {get_model_size_mb(model):.2f} MB")
    
    # Create dummy input
    dummy_input = torch.randn(1, 40, 157)  # MFCC shape
    
    # Test forward pass
    with torch.no_grad():
        output = model(dummy_input)
    print(f"🧪 Forward pass test: {dummy_input.shape} -> {output.shape}")
    
    # Convert to requested formats
    base_name = f"{args.model}_baby_cry_model"
    
    if 'onnx' in args.formats:
        print("\n🔄 Converting to ONNX...")
        onnx_path = f"{args.output_dir}/{base_name}.onnx"
        
        convert_to_onnx(
            model=model,
            dummy_input=dummy_input,
            output_path=onnx_path,
            input_names=['mfcc_features'],
            output_names=['cry_predictions']
        )
        
        # Check ONNX model size
        onnx_size_kb = Path(onnx_path).stat().st_size / 1024
        print(f"📏 ONNX model size: {onnx_size_kb:.1f} KB")
    
    if 'tflite' in args.formats:
        print("\n🔄 Converting to TensorFlow Lite...")
        if 'onnx' not in args.formats:
            # Need ONNX as intermediate format
            onnx_path = f"{args.output_dir}/{base_name}.onnx"
            convert_to_onnx(
                model=model,
                dummy_input=dummy_input,
                output_path=onnx_path,
                input_names=['mfcc_features'],
                output_names=['cry_predictions']
            )
        
        tflite_path = f"{args.output_dir}/{base_name}.tflite"
        success = convert_to_tflite(onnx_path, tflite_path)
        
        if success and Path(tflite_path).exists():
            tflite_size_kb = Path(tflite_path).stat().st_size / 1024
            print(f"📏 TFLite model size: {tflite_size_kb:.1f} KB")
    
    print("\n✅ Model conversion completed!")
    print(f"📁 Output files saved to: {args.output_dir}")
    
    # Create model info file
    model_info = {
        'model_type': args.model,
        'input_shape': [1, 40, 157],
        'output_shape': [1, 5],
        'class_names': ['hunger', 'pain', 'discomfort', 'tired', 'normal'],
        'parameters': sum(p.numel() for p in model.parameters()),
        'pytorch_size_mb': get_model_size_mb(model)
    }
    
    import json
    with open(f"{args.output_dir}/model_info.json", 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("📝 Model info saved to model_info.json")

if __name__ == "__main__":
    main()