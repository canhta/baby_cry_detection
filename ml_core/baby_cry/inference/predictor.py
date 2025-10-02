"""
Inference Module for Baby Cry Detection
Handles model loading and prediction
"""

import torch
import torch.nn.functional as F
import librosa
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

class BabyCryPredictor:
    """
    Production-ready predictor for baby cry detection
    """
    
    CRY_TYPES = ['hunger', 'pain', 'discomfort', 'tired', 'normal']
    CRY_DESCRIPTIONS = {
        'hunger': 'Baby is hungry and needs feeding',
        'pain': 'Baby is experiencing pain or discomfort', 
        'discomfort': 'Baby feels uncomfortable (wet diaper, too hot/cold)',
        'tired': 'Baby is sleepy and needs rest',
        'normal': 'Baby is content or making normal sounds'
    }
    
    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize predictor with trained model
        
        Args:
            model: Trained PyTorch model
            device: Device to run inference on
        """
        self.device = device
        self.model = model.to(device)
        self.model.eval()
        
        # Audio processing parameters
        self.sr = 16000
        self.duration = 5.0
        self.n_mfcc = 40
        self.n_samples = int(self.sr * self.duration)
        
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, model_class, device: str = 'cpu'):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_class: Model class to instantiate
            device: Device to load model on
            
        Returns:
            BabyCryPredictor instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = model_class(n_classes=5)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return cls(model, device)
    
    def preprocess_audio(self, audio_path: str) -> torch.Tensor:
        """
        Preprocess audio file for inference
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed MFCC features
        """
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Pad or trim to exact length
        if len(audio) < self.n_samples:
            audio = np.pad(audio, (0, self.n_samples - len(audio)))
        else:
            audio = audio[:self.n_samples]
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            hop_length=512,
            n_fft=2048
        )
        
        # Normalize features
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        return torch.FloatTensor(mfcc).unsqueeze(0)
    
    def preprocess_audio_array(self, audio_array: np.ndarray) -> torch.Tensor:
        """
        Preprocess audio array for real-time inference
        
        Args:
            audio_array: Raw audio samples
            
        Returns:
            Preprocessed MFCC features
        """
        # Ensure correct length
        if len(audio_array) < self.n_samples:
            audio_array = np.pad(audio_array, (0, self.n_samples - len(audio_array)))
        else:
            audio_array = audio_array[:self.n_samples]
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio_array, 
            sr=self.sr, 
            n_mfcc=self.n_mfcc,
            hop_length=512,
            n_fft=2048
        )
        
        # Normalize
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
        
        return torch.FloatTensor(mfcc).unsqueeze(0)
    
    def predict(self, audio_input) -> Dict[str, Any]:
        """
        Predict cry type from audio
        
        Args:
            audio_input: Either file path (str) or audio array (np.ndarray)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Preprocess audio
            if isinstance(audio_input, str):
                mfcc_features = self.preprocess_audio(audio_input)
            elif isinstance(audio_input, np.ndarray):
                mfcc_features = self.preprocess_audio_array(audio_input)
            else:
                raise ValueError("Audio input must be file path (str) or audio array (np.ndarray)")
            
            mfcc_features = mfcc_features.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(mfcc_features)
                probabilities = F.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
            
            result = {
                'cry_type': self.CRY_TYPES[predicted_idx],
                'confidence': confidence,
                'description': self.CRY_DESCRIPTIONS[self.CRY_TYPES[predicted_idx]],
                'all_probabilities': {
                    cry_type: prob.item() 
                    for cry_type, prob in zip(self.CRY_TYPES, probabilities[0])
                },
                'prediction_index': predicted_idx
            }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'cry_type': 'unknown',
                'confidence': 0.0,
                'description': 'Error occurred during prediction'
            }
    
    def predict_batch(self, audio_inputs: list) -> list:
        """
        Predict cry types for batch of audio inputs
        
        Args:
            audio_inputs: List of audio file paths or arrays
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for audio_input in audio_inputs:
            result = self.predict(audio_input)
            results.append(result)
        return results


def predict_baby_cry(model: torch.nn.Module, audio_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Simple prediction function for backward compatibility
    
    Args:
        model: Trained PyTorch model
        audio_path: Path to audio file
        device: Device to run inference on
        
    Returns:
        Dictionary with prediction results
    """
    predictor = BabyCryPredictor(model, device)
    return predictor.predict(audio_path)