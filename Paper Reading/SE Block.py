import torch
import torch.nn.functional as F
import torch.nn as nn

class SEBlock(nn.Module):
    
    def __init__(self, input_channels: int, reduction_ratio: int = 16):
                
        self.bottleneck = input_channels / reduction_ratio
        
        self.excitation = nn.Sequential(
            nn.Linear(in_features = input_channels, 
                      out_features = self.bottleneck),
            nn.ReLU,
            nn.Linear(in_features = self.bottleneck,
                      out_features = input_channels)
            )
        
    def forward(self, x):
        
        x1 = F.adaptive_avg_pool2d(x, 1)
        x1 = self.excitation(x)
        x1 = F.sigmoid(x)
        
        return x * x1