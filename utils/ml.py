from torch import nn

def build_mlp(sizes, activation=nn.ReLU, final_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif final_activation is not None:
            layers.append(final_activation(dim=-1))
    return nn.Sequential(*layers)
