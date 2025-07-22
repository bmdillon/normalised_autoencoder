import torch
import torch.nn.functional as F
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, max_size=10000, replay_ratio=0.95):
        self.buffer = []
        self.max_size = max_size
        self.replay_ratio = replay_ratio

    def push(self, samples):
        samples = samples.detach().cpu()
        for sample in samples:
            self.buffer.append(sample)
            if len(self.buffer) > self.max_size:
                self.buffer.pop(0)

    def sample(self, n):
        if len(self.buffer) < n:
            return None
        return torch.stack(random.choices(self.buffer, k=n))

def langevin_dynamics(x, energy_fn, steps=20, step_size=1e-2, temperature=1.0, clip_grad=None, mh=False):
    x = x.clone().detach().requires_grad_(True)
    noise_scale =  np.sqrt(temperature * 2 * step_size)
    initial_energy = energy_fn(x).detach() if mh else None

    for _ in range(steps):
        energy = energy_fn(x).mean()
        grad = torch.autograd.grad(energy, x, create_graph=False)[0]
        if clip_grad:
            grad = torch.clamp(grad, -clip_grad, clip_grad)
        x.data -= 0.5 * step_size * grad
        x.data += noise_scale * torch.randn_like(x)

    if mh and initial_energy is not None:
        new_energy = energy_fn(x).detach()
        accept_ratio = torch.exp(initial_energy - new_energy)
        accept = torch.rand_like(accept_ratio) < accept_ratio
        x = torch.where(accept[:, None], x, x.detach())

    return x.detach()
