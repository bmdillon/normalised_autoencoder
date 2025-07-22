import os
import argparse
import torch
import yaml
from models.nae import NormalizedAutoEncoder
from training.trainer import Trainer
from utils.data import load_data, load_val


def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NormalizedAutoEncoder(
        input_dim=cfg['model']['input_dim'],
        latent_dim=cfg['model']['latent_dim'],
        encoder_dims=cfg['model'].get('encoder_dims', [64, 32]),
        decoder_dims=cfg['model'].get('decoder_dims', [32, 64]),
        buffer_size=cfg['training'].get('buffer_size', 10000),
        replay_ratio=cfg['training'].get('replay_ratio', 0.95),

        # energy func
        energy_func=cfg['energy'].get('energy_func'),
        ot_method=cfg['energy'].get('ot_method'),
        blur=cfg['energy'].get('blur'),
        scaling=cfg['energy'].get('scaling'),
        p=cfg['energy'].get('p'),

        # langevin in x
        langevin_steps=cfg['training'].get('langevin_steps', 20),
        langevin_step_size=cfg['training'].get('langevin_step_size', 1e-2),
        clip_grad=cfg['training'].get('clip_grad', None),
        mh=cfg['training'].get('mh', False),
        temperature_x=cfg['training'].get('temperature_x', 1.0),

        # langevin in z
        langevin_z_steps=cfg['training'].get('langevin_z_steps', 20),
        langevin_z_step_size=cfg['training'].get('langevin_z_step_size', 1e-2),
        clip_grad_z=cfg['training'].get('clip_grad_z', None),
        mh_z=cfg['training'].get('mh_z', False),
        temperature_z=cfg['training'].get('temperature_z', 1.0),

    )

    trainer = Trainer(model, device, ae_lr=cfg['training']['ae_lr'], nae_lr=cfg['training']['nae_lr'], logfile = cfg['save'].get('logfile'))
    dataloader = load_data(cfg['data']['path'], cfg['data']['batch_size'], gfilter=cfg['data']['gaussian_filter'], gfsigma=cfg['data']['gfsigma'])
    val_loader = load_val(cfg['data']['val_path'], cfg['data']['batch_size'], gfilter=cfg['data']['gaussian_filter'], gfsigma=cfg['data']['gfsigma'])
    
    trainer.train_ae(dataloader, val_loader, cfg['training'].get('eval_every', 10), cfg['training']['ae_epochs'])
    trainer.train_nae(dataloader, val_loader, cfg['training'].get('eval_every', 10), cfg['training']['nae_epochs'])
    os.makedirs(os.path.dirname(cfg['save'].get('net_path')), exist_ok=True)
    torch.save( model.state_dict(), cfg['save'].get('net_path') )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args = parser.parse_args()
    main(args.config)
