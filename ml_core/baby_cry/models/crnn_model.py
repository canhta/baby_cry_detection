"""
CRNN Model for Baby Cry Detection
Combines CNN spatial feature extraction with LSTM temporal modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN_BabyCry(nn.Module):
    """
    CRNN (CNN + LSTM) for baby cry detection
    Combines spatial feature extraction with temporal modeling
    Best overall performance but heavier for mobile
    """
    def __init__(self, n_classes: int = 5):
        super(CRNN_BabyCry, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        
        # LSTM layers
        self.lstm = nn.LSTM(256, 128, num_layers=2, 
                           batch_first=True, bidirectional=True)
        
        # Fully connected
        self.fc = nn.Linear(128 * 2, n_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, n_mfcc, time_steps)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Prepare for LSTM: (batch, channels, freq, time) -> (batch, time, features)
        x = x.permute(0, 3, 1, 2)
        batch_size, time_steps, channels, freq = x.size()
        x = x.contiguous().view(batch_size, time_steps, -1)
        
        # LSTM temporal modeling
        x, _ = self.lstm(x)
        
        # Global average pooling over time
        x = torch.mean(x, dim=1)
        
        # Classification
        x = self.dropout(x)
        x = self.fc(x)
        
        return x