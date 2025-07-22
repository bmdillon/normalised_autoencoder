import torch
import torch.nn as nn
from utils.ml import build_mlp
from utils.ml import NormActivation

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=1600, latent_dim=3, encoder_dims=[128,64], decoder_dims=[64,128] ):
        super().__init__()
        self.encoder = build_mlp( [input_dim] + encoder_dims + [latent_dim], activation=nn.PReLU, final_activation=NormActivation() )
        self.decoder = build_mlp( [latent_dim] + decoder_dims + [input_dim], activation=nn.PReLU, final_activation=nn.Softmax(dim=-1) )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)
