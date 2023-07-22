
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


gen_var = lambda x, y : Variable(x, requires_grad=y)

class EtaWeights(nn.Module):
    def __init__(self, eta : float, device = None, min=np.log(2), max=10):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.register_parameter(
            'eta_value',
            torch.nn.Parameter(
                torch.Tensor([eta]).to(self.device),
                requires_grad=True))
        self.clamp = lambda x : torch.clamp(x, min=min, max=max)
        self.eta =  torch.tensor([eta], requires_grad=True).to(device)
        self.mask = torch.Tensor([0.], requires_grad=False).to(self.device)

        self.weighting = lambda x, y : (-x/y) + 1
    
    def set_mask(self, mask : float):
        self.mask = torch.Tensor([mask], requires_grad=False).to(self.device)

    def no_grad(self, loss, eta):
        with torch.no_grad():
            weight = gen_var(torch.zeros(loss.size()), True).to(self.device)
            for i in range(len(loss)):
                if loss[i] > eta:
                    weight[i] = loss[i] * self.mask * eta
                else:
                    weight[i]  = self.weighting(loss[i], eta)
            return weight
    
    def forward(self, loss=None):
        weight = gen_var(torch.zeros(loss.size()), True).to(self.device)

        for i in range(len(loss)):
            if loss[i] > self.eta:
                weight[i] = loss[i] * self.mask.requires_grad_() * self.eta
            else:
                weight[i] = self.weighting(loss[i], self.eta)

        return weight
    
class Weights(nn.Module):
    def __init__(self, shape, device = None):
        super().__init__()
        self.shape = shape
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.nn.Parameter(torch.ones(shape).to(self.device), requires_grad=False)

    def set_weight(self, idx, val):
        self.weight[idx] = nn.functional.sigmoid(val).to(self.device)

    def __getitem__(self, idx):
        return gen_var(self.weight[idx], True).to(self.device)

    def forward(self, loss=None, idx=0):
        return gen_var(self.weight[idx], True).to(self.device)