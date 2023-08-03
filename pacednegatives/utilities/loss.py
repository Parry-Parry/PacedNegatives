import torch.nn.functional as F
import torch

def LCEcrossentropy(plogits, nlogits, op, on, weights):
    pce = F.cross_entropy(plogits.view(-1, plogits.size(-1)), op.view(-1), reduction='none')
    nce = F.cross_entropy(nlogits.view(-1, nlogits.size(-1)), on.view(-1), reduction='none')

    nce = nce.view(-1, nlogits.size(-2), nce.size(-1))
    # nce must become 2 dimensional
    ce = pce + nce
    v = weights.no_grad(ce, weights.eta)
    loss = torch.mean(ce * v)
    return loss
