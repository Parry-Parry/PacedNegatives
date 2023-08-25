import torch
import torch.nn.functional as F

margin = lambda x : x[::2] - x[1::2]

def MarginMSELoss(x, y):
    student_margin = margin(x)
    losses = [F.mse_loss(student_margin, margin(y[:, i]))for i in range(y.shape[-1])]
    return torch.mean(torch.stack(losses))