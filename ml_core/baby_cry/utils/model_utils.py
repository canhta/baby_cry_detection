"""
Model Utilities
Functions for model management, saving, loading, and conversion
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from typing import Dict, Any, Optional, Tuple

def save_model(model: nn.Module, save_path: str, 
               optimizer: Optional[torch.optim.Optimizer] = None,
               epoch: Optional[int] = None,
               metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Save model checkpoint with metadata
    
    Args:
        model: PyTorch model to save
        save_path: Path to save checkpoint
        optimizer: Optimizer state (optional)
        epoch: Current epoch (optional) 
        metrics: Training metrics (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'model_config': getattr(model, 'config', {}),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
        
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"✅ Model saved to: {save_path}")


def load_model(model_class, checkpoint_path: str, device: str = 'cpu') -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint
    
    Args:
        model_class: Model class to instantiate
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Tuple of (model, checkpoint_metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', None),
        'metrics': checkpoint.get('metrics', {}),
        'model_class': checkpoint.get('model_class', 'Unknown')
    }
    
    print(f"✅ Model loaded from: {checkpoint_path}")
    return model, metadata


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count parameters in model
    
    Args:
        model: PyTorch model
        trainable_only: Count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in MB
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in megabytes
    """
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def convert_to_onnx(model: nn.Module, dummy_input: torch.Tensor, 
                   output_path: str, input_names: list = None, 
                   output_names: list = None) -> None:
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model: PyTorch model to convert
        dummy_input: Sample input tensor
        output_path: Path to save ONNX model
        input_names: Names for input tensors
        output_names: Names for output tensors
    """
    model.eval()
    
    # Default names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Create directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            input_names[0]: {0: 'batch_size'},
            output_names[0]: {0: 'batch_size'}
        }
    )
    
    print(f"✅ Model exported to ONNX: {output_path}")


def print_model_summary(model: nn.Module, input_size: tuple) -> None:
    """
    Print model architecture summary
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (without batch dimension)
    """
    print(f"\n📋 Model Summary: {model.__class__.__name__}")
    print("=" * 60)
    
    # Count parameters
    trainable_params = count_parameters(model, trainable_only=True)
    total_params = count_parameters(model, trainable_only=False)
    model_size = get_model_size_mb(model)
    
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {model_size:.2f} MB")
    
    # Test forward pass
    dummy_input = torch.randn(1, *input_size)
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
    
    print("=" * 60)