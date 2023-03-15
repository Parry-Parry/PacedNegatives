import torch
import torch.nn as nn
import numpy as np

class SPL(nn.Module):
    def __init__(self, loss_func, k=1000, decay=1.1) -> None:
        super().__init__()
        self.func = loss_func
        self.k = k
        self.decay = decay
    
    def forward(self, *args, **kwargs):
        v = kwargs.pop('v', np.ones(self.b))
        loss = self.func(*args, **kwargs) - 1/self.k * np.sum(v) 
        self.k = self.k * self.decay 
        return loss
