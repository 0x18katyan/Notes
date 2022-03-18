import torch
import torch.nn.functional as F
import torch.nn as nn

class SEBlock(nn.Module):
    
    def __init__(self, input_channels: int, reduction_ratio: int = 16):
        
        super(SEBlock, self).__init__()
        
        self.bottleneck = input_channels / reduction_ratio
        
        self.excitation = nn.Sequential(
            nn.Linear(in_features = input_channels, 
                      out_features = self.bottleneck,
                      bias = False),
            nn.ReLU,
            nn.Linear(in_features = self.bottleneck,
                      out_features = input_channels,
                      bias = False)
            )
        
    def forward(self, x):
        
        ## Squeeze Module
        x1 = F.adaptive_avg_pool2d(x, 1)
        
        ## Excitation Module
        x1 = self.excitation(x)
        
        ## Scale Module
        x1 = F.sigmoid(x)
        x = x * x1
        
        return x