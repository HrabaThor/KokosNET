import numpy as np
import torch

def to_tensor(x):
    """Convert array into pytorch tensor"""
    x = np.array(x)
    return torch.from_numpy(x).float()
