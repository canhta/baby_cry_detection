#!/usr/bin/env python3
"""
Baby Cry Detection - Simple CLI Entry Point
Usage:
  python cli.py train --model mobile --epochs 50
  python cli.py predict --audio test.wav --model models/best_model.pth  
  python cli.py convert --checkpoint models/best_model.pth --model mobile
"""

import sys
import os

# Add ml_core to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ml_core'))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("🍼 Baby Cry Detection CLI")
        print("\nUsage:")
        print("  python baby_cry.py train --model mobile --epochs 50")
        print("  python baby_cry.py predict --audio test.wav --model models/best_model.pth")
        print("  python baby_cry.py convert --checkpoint models/best_model.pth --model mobile")
        print("\nCommands:")
        print("  train    - Train a model")
        print("  predict  - Make predictions on audio file") 
        print("  convert  - Convert model for mobile deployment")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "train":
        from baby_cry.cli.train import main
        # Remove 'train' from args
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    elif command == "predict":
        from baby_cry.cli.predict import main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    elif command == "convert":
        from baby_cry.cli.convert_model import main
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        main()
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: train, predict, convert")
        sys.exit(1)