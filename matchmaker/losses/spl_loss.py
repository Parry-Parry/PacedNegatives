import torch
import torch.nn as nn
import numpy as np

class SPL(nn.Module):
    def __init__(self, loss_func, K) -> None:
        super().__init__()
        self.func = loss_func
        self.K = K 
    
    def forward(self, *args, **kwargs):
        v = kwargs.pop('v', np.zeros(1))
        return self.func(*args, **kwargs) - 1/self.K * np.sum(v) 