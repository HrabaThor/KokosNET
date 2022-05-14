import torch
from torch import nn
import numpy as np

def to_tensor(x):
    """Convert array into pytorch tensor"""
    x = np.array(x)
    return torch.from_numpy(x).float()


class FlexibleDenseNet(nn.Module):
    def __init__(self, layers=[64,32,16,8,4], activations=nn.ReLU, last_a=None):
        '''Initialize flexible (by list of dimensions defined) densely connected network module.'''
        super().__init__()
        # Safety first!
        assert len(layers) > 1
        # Save the model structure
        self.model = nn.ModuleList()
        # Plan the activations - will be an activation function or None in case of no activation layer wanted
        activ = ([activations] * (len(layers) - 2))
        activ.append(last_a)
        # Create layers
        for i, layer in enumerate(layers[1:]):
            self.model.append(nn.Linear(layers[i], layers[i+1]))
            if activ[i]:
                self.model.append(activ[i]())
        
    def forward(self, x):
        # Iterate through each layer of model, return the last value
        for layer in self.model:
            x = layer(x)
        return x