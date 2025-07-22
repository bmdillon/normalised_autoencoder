import torch
import torch.nn as nn
import torch.nn.functional as F

def build_mlp(sizes, activation=nn.ReLU, final_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)

class NormActivation(nn.Module):
    def __init__(self, p=2, dim=1, eps=1e-12):
        super(NormActivation, self).__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


