import torch
from sklearn.metrics import roc_auc_score

def evaluate(model, val_loader, device, logger):
    logger.info( 'eval' )
    model.eval()
    energies, labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            energy = model.energy(x)
            energies.extend(energy.cpu().tolist())
            labels.extend(y.cpu().tolist())
    auc = roc_auc_score(labels, energies)
    logger.info( f"[eval] AUC: {auc:.4f}" )
    return auc
