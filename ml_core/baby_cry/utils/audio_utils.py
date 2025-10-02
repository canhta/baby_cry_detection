"""
Audio Processing Utilities
Common functions for audio loading and feature extraction
"""

import librosa
import numpy as np
import torch
from typing import Tuple, Optional

def load_audio(audio_path: str, sr: int = 16000, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """
    Load audio file with specified sample rate and duration
    
    Args:
        audio_path: Path to audio file
        sr: Target sample rate
        duration: Duration in seconds (None for full file)
        
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, loaded_sr = librosa.load(audio_path, sr=sr, duration=duration)
    return audio, loaded_sr


def normalize_audio(audio: np.ndarray, method: str = 'peak') -> np.ndarray:
    """
    Normalize audio signal
    
    Args:
        audio: Audio signal array
        method: Normalization method ('peak', 'rms')
        
    Returns:
        Normalized audio array
    """
    if method == 'peak':
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
    elif method == 'rms':
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            audio = audio / rms
    
    return audio


def extract_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 40, 
                hop_length: int = 512, n_fft: int = 2048) -> np.ndarray:
    """
    Extract MFCC features from audio
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        hop_length: Hop length for STFT
        n_fft: FFT window size
        
    Returns:
        MFCC feature matrix
    """
    mfcc = librosa.feature.mfcc(
        y=audio, 
        sr=sr, 
        n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=n_fft
    )
    
    # Normalize features
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    
    return mfcc


def extract_mel_spectrogram(audio: np.ndarray, sr: int = 16000, n_mels: int = 128) -> np.ndarray:
    """
    Extract mel spectrogram from audio
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        n_mels: Number of mel bands
        
    Returns:
        Mel spectrogram in dB scale
    """
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def pad_or_trim_audio(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Pad or trim audio to target length
    
    Args:
        audio: Audio signal array
        target_length: Target length in samples
        
    Returns:
        Audio array with target length
    """
    if len(audio) < target_length:
        # Pad with zeros
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        # Trim to target length
        audio = audio[:target_length]
    
    return audio