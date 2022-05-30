"""
    File: nets.py 
    Project: KNN - Actor-Critic 
    Authors: Martin Kostelník (xkoste12), Michal Glos (xglosm01), Michal Szymik (xszymi00) 
    Date: 2022-05-30

    Description: In this file are the classes for actor and critic nets.

"""

from torch import nn
import torch

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
        for i in range(len(layers)-1):
            self.model.append(nn.Linear(layers[i], layers[i+1]))
            if activ[i]:
                # Having softmax with dynamic output size was deprecated
                self.model.append(activ[i]() if not activ[i] == nn.Softmax else activ[i](layers[i+1]))
        
    def forward(self, x):
        # Iterate through each layer of model, return the last value
        for layer in self.model:
            x = layer(x)
        return x

class Critic(nn.Module):
    def __init__(self, in_shape, out_shape, layers=[400, 300], name='critic'):
        '''Critic neural network (fully connected'''
        super().__init__()
        # Save the name for further use with checkpoints
        self.name = name
        # Define neural network model
        self.model = FlexibleDenseNet(layers=[in_shape + out_shape] + layers + [out_shape])

    def forward(self, x):
        return self.model(x)

class Actor(nn.Module):
    def __init__(self, in_shape, out_shape, layers=[400, 300], name='actor', max_action=2):
        '''Actor neural network (fully connected)'''
        super().__init__()
        # Coefficient for output normalisation in required interval
        self.max_output = max_action
        # Save the name for further use with checkpoints
        self.name = name
        # Define neural network model
        self.model = FlexibleDenseNet(layers=[in_shape] + layers + [out_shape], last_a=nn.Tanh)

    def forward(self, x):
        return self.model(x) * self.max_output