import torch

def init_LCEcrossentropy(ignore_index=-100):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    def LCEcrossentropy(plogits, nlogits, op, on, weights=None, backward=True):
        pce = loss_fn(plogits.view(-1, plogits.size(-1)), op.view(-1))
        nce = loss_fn(nlogits.view(-1, nlogits.size(-1)), on.view(-1))

        nce = nce.view(-1, nlogits.size(-2))
        nce = torch.mean(nce, dim=1)
        # nce must become 2 dimensional
        ce = pce + nce
        if weights is not None:
            if backward:
                v = weights.forward(ce)
            else:
                v = weights.no_grad(ce, weights.eta)
            loss = torch.mean(ce * v)
        return loss
    return LCEcrossentropy