"""
Training Module for Baby Cry Detection Models
Handles training, validation, and model checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import json
from tqdm import tqdm

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader,
                epochs: int = 100,
                lr: float = 0.001,
                device: str = 'cuda',
                save_path: str = 'models/checkpoints',
                early_stopping_patience: int = 15) -> Dict[str, Any]:
    """
    Training function for baby cry detection models
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ('cuda' or 'cpu')
        save_path: Path to save model checkpoints
        early_stopping_patience: Epochs to wait before early stopping
        
    Returns:
        Dictionary with training history and metrics
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Create save directory
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"Training {model.__class__.__name__} on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("=" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.squeeze().to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Update progress bar
            train_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{train_acc:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Handle empty validation set
        if len(val_loader) > 0:
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for data, target in val_pbar:
                    data, target = data.to(device), target.squeeze().to(device)
                    output = model(data)
                    loss = criterion(output, target)
                    
                    val_loss += loss.item()
                    pred = output.argmax(dim=1)
                    val_correct += pred.eq(target).sum().item()
                    val_total += target.size(0)
                    
                    val_acc = 100. * val_correct / val_total
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{val_acc:.2f}%'
                    })
        
        # Calculate epoch metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100. * train_correct / train_total
        
        # Handle validation metrics for empty validation set
        if len(val_loader) > 0:
            epoch_val_loss = val_loss / len(val_loader)
            epoch_val_acc = 100. * val_correct / val_total
        else:
            epoch_val_loss = epoch_train_loss  # Use training loss as fallback
            epoch_val_acc = epoch_train_acc    # Use training acc as fallback
            print("⚠️  No validation data available, using training metrics for validation")
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        
        print(f'Epoch [{epoch+1:3d}/{epochs}] '
              f'Train Loss: {epoch_train_loss:.4f} '
              f'Train Acc: {epoch_train_acc:.2f}% '
              f'Val Loss: {epoch_val_loss:.4f} '
              f'Val Acc: {epoch_val_acc:.2f}% '
              f'LR: {current_lr:.2e}')
        
        # Save best model (use validation acc if available, otherwise training acc)
        current_metric = epoch_val_acc if len(val_loader) > 0 else epoch_train_acc
        if current_metric > best_val_acc:
            best_val_acc = current_metric
            patience_counter = 0
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': epoch_val_loss,
                'history': history
            }
            
            model_name = model.__class__.__name__.lower()
            torch.save(checkpoint, f'{save_path}/best_{model_name}.pth')
            torch.save(model.state_dict(), f'{save_path}/best_{model_name}_weights.pth')
            
            print(f'✅ Best model saved! Val Acc: {best_val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping after {epoch + 1} epochs')
            break
        
        print('-' * 60)
    
    # Save final training history
    with open(f'{save_path}/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print(f"🎯 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"📁 Model saved to: {save_path}")
    
    return {
        'model': model,
        'best_val_acc': best_val_acc,
        'history': history,
        'save_path': save_path
    }


def validate_model(model: nn.Module, 
                  test_loader: DataLoader, 
                  device: str = 'cuda') -> Dict[str, float]:
    """
    Validate model on test set
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader  
        device: Device to run validation on
        
    Returns:
        Dictionary with test metrics
    """
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for data, target in test_pbar:
            data, target = data.to(device), target.squeeze().to(device)
            output = model(data)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            pred = output.argmax(dim=1)
            test_correct += pred.eq(target).sum().item()
            test_total += target.size(0)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            test_acc = 100. * test_correct / test_total
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{test_acc:.2f}%'
            })
    
    test_loss /= len(test_loader)
    test_acc = 100. * test_correct / test_total
    
    print(f'\n📊 Test Results:')
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'predictions': all_predictions,
        'targets': all_targets
    }