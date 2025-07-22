import torch
from geomloss import SamplesLoss

def ot_loss( x, x_hat, blur=1.0, scaling=0.9, p=2.0, method="gaussian", backend="auto" ):
    """
    Computes per-image OT loss using geomloss.
    
    Args:
        x, x_hat: tensors of shape (B, 1600)
        blur: entropic regularization (blur=sqrt(eps) in geomloss)
    
    Returns:
        Tensor of shape (B,) — OT losses per image

    SamplesLoss details:
    https://www.kernel-operations.io/geomloss/_modules/geomloss/samples_loss.html#SamplesLoss
    """
    B, D = x.shape
    device = x.device

    sinkhorn = SamplesLoss(method, blur=blur, scaling=scaling, p=p, backend=backend)

    i = j = torch.linspace(0, 1, 40)
    I, J = torch.meshgrid(i, j, indexing='ij')
    pos = torch.stack([I.flatten(), J.flatten()], dim=1)
    pos_batch = pos.unsqueeze(0).expand(B, -1, -1) 

    loss = sinkhorn(x, pos_batch, x_hat, pos_batch)

    return loss
