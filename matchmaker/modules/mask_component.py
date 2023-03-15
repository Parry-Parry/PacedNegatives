import torch
import torch.nn as nn 
from math import ceil

class Mask(nn.Module):
    i = None
    b = None
    def __init__(self, n, b, dtype=torch.fp16) -> None:
        super().__init__()
        self.i = ceil(n/b)
        self.b = b
        self.mask = nn.parameter(torch.ones((self.i, b), dtype=dtype))
    
    def return_mask(self, idx):
        x = torch.zeros((self.i, self.b))
        x[idx] = torch.ones(self.b)
        return x

    def forward(self, idx):
        return torch.sigmoid(self.mask).cpu().detach().numpy()[idx]
    
    def parameters(self):
        return self.mask