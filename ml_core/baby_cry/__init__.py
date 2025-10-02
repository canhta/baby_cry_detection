"""
Baby Cry Detection - ML Core Package
Production-ready machine learning models and utilities
"""

__version__ = "1.0.0"
__author__ = "Baby Cry Detection Team"

from .models.mobile_model import MobileCNN_BabyCry, MobileAudioProcessor, MobileBabyCryPredictor
from .models.cnn_model import CNN_BabyCry
from .models.crnn_model import CRNN_BabyCry
from .data.dataset import BabyCryDataset
from .training.trainer import train_model
from .inference.predictor import predict_baby_cry

__all__ = [
    "MobileCNN_BabyCry",
    "MobileAudioProcessor", 
    "MobileBabyCryPredictor",
    "CNN_BabyCry",
    "CRNN_BabyCry",
    "BabyCryDataset",
    "train_model",
    "predict_baby_cry"
]