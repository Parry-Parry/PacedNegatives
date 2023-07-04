
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


gen_var = lambda x, y : Variable(x, requires_grad=y)

class EtaWeights(nn.Module):
    def __init__(self, eta : float, shape, batch_size, device = None, min=np.log(2), max=10):
        super().__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.register_parameter(
            'eta_value',
            torch.nn.Parameter(
                torch.Tensor([eta]).to(self.device),
                requires_grad=True))
        self.clamp = lambda x : torch.clamp(x, min=min, max=max)
        self.eta =  torch.tensor([eta], requires_grad=True).to(device)

        self.weighting = lambda x, y : nn.functional.sigmoid((-x/y) + 1)
        self.weights = torch.nn.Parameter(torch.ones(shape).to(self.device), requires_grad=False)

    def no_grad(self, loss, eta):
        with torch.no_grad():
            weight = gen_var(torch.zeros(loss.size()), True).to(self.device)
            for i in range(len(loss)):
                if loss[i] > eta:
                    pass
                else:
                    weight[i]  = self.weighting(loss[i], eta)
            return torch.nn.functional.sigmoid(weight)
    
    def forward(self, loss=None, idx=None):
        if loss is None:
            assert idx is not None
            return self.weights[idx]
        
        weight = gen_var(torch.zeros(loss.size()), True).to(self.device)

        for i in range(len(loss)):
            if loss[i] > self.eta:
                weight[i] = torch.zeros(1).to(self.device).requires_grad_() * self.eta
            else:
                weight[i] = self.weighting(loss[i], self.eta)
        
        if idx is not None: self.weights[idx] = weight
        return torch.nn.functional.sigmoid(weight)
    
class Weights(nn.Module):
    def __init__(self, shape, device = None):
        super().__init__()
        self.shape = shape
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight = torch.nn.Parameter(torch.ones(shape).to(self.device), requires_grad=True)

    def set_weight(self, idx, val):
        self.weight[idx] = nn.functional.sigmoid(val)

    def __getitem__(self, idx):
        return self.weight[idx]

    def forward(self, loss=None, idx=0):
        return self.weight[idx]