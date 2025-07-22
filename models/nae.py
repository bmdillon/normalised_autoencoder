
import torch
import torch.nn as nn
from models.autoencoder import AutoEncoder
from models.sampling import ReplayBuffer, langevin_dynamics
from utils.losses import ot_loss

class NormalisedAutoEncoder(nn.Module):
    def __init__(self, input_dim=1600, latent_dim=3, encoder_dims=[128,64], decoder_dims=[64,128], 
                 energy_func='mse', ot_method='gaussian', blur=1.0, scaling=0.9, p=2.0,
                 buffer_size=10000, replay_ratio=0.95,
                 langevin_steps=20, langevin_step_size=1e-2, clip_grad=None, mh=False,
                 langevin_z_steps=20, langevin_z_step_size=1e-2, clip_grad_z=None, mh_z=False,
                 temperature_x=1.0, temperature_z=1.0):
        super().__init__()
        self.autoencoder = AutoEncoder(input_dim, latent_dim, encoder_dims, decoder_dims)
        self.buffer = ReplayBuffer(buffer_size, replay_ratio)

        self.energy_func = energy_func
        self.ot_method = ot_method
        self.blur = blur
        self.scaling = scaling
        self.p = p

        # Input space Langevin
        self.langevin_steps = langevin_steps
        self.langevin_step_size = langevin_step_size
        self.clip_grad = clip_grad
        self.mh = mh
        self.temperature_x = temperature_x

        # Latent space Langevin
        self.langevin_z_steps = langevin_z_steps
        self.langevin_z_step_size = langevin_z_step_size
        self.clip_grad_z = clip_grad_z
        self.mh_z = mh_z
        self.temperature_z = temperature_z

        self.latent_dim = latent_dim

    def forward(self, x):
        return self.autoencoder(x)

    def energy(self, x):
        if self.energy_func == 'mse':
            return ((x - self.autoencoder(x)) ** 2).mean(dim=1)
        elif self.energy_func == 'ase':
            return torch.abs((x - self.autoencoder(x))).mean(dim=1)
        elif self.energy_func == 'ot':
            return ot_loss( x, self.autoencoder(x), blur=self.blur, scaling=self.scaling, p=self.p, method=self.ot_method )
        else:
            return None

    def sample_negative(self, batch_size):
        device = next(self.parameters()).device
        z = torch.randn(batch_size, self.latent_dim).to(device)

        # Langevin in latent space
        z = langevin_dynamics(
            z, lambda z_: self.energy(self.autoencoder.decode(z_)),
            steps=self.langevin_z_steps,
            step_size=self.langevin_z_step_size,
            clip_grad=self.clip_grad_z,
            mh=self.mh_z,
            temperature=self.temperature_z
        )

        x_init = self.autoencoder.decode(z)

        # Replay buffer usage
        if self.buffer.sample(batch_size) is not None:
            use_replay = torch.rand(batch_size) < self.buffer.replay_ratio
            replay_samples = self.buffer.sample(batch_size)
            x_init[use_replay] = replay_samples[use_replay]

        x_sampled = langevin_dynamics(
            x_init, self.energy,
            steps=self.langevin_steps,
            step_size=self.langevin_step_size,
            clip_grad=self.clip_grad,
            mh=self.mh,
            temperature=self.temperature_x
        )

        x_sampled = torch.clamp(x_sampled, min=0.0)
        x_sampled = x_sampled / torch.sum(x_sampled, -1, keepdim=True)

        self.buffer.push(x_sampled)
        return x_sampled
