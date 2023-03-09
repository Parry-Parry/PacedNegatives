import torch
import torch.nn as nn 
from math import ceil

class Mask(nn.Module):
    def __init__(self, n, b, dtype=torch.fp16) -> None:
        super().__init__()
        self.masks = [nn.parameter(torch.ones(b, dtype=dtype)) for i in range(ceil(n / b))]

    def parameters(self):
        return self.masks