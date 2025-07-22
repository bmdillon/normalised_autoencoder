import torch
from tqdm import tqdm
from utils.eval import evaluate
from utils.logger import get_logger
import uuid

class Trainer:
    def __init__(self, model, device, ae_lr=1.0e-3, nae_lr=1.0e-4, logfile=None):
        self.model = model.to(device)
        self.device = device
        self.ae_opt = torch.optim.Adam(model.parameters(), lr=ae_lr)
        self.nae_opt = torch.optim.Adam(model.parameters(), lr=nae_lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if logfile == None:
            logfile = f'logs/log_{uuid.uuid4()}.txt'
        self.logger = get_logger(__name__, logfile=logfile)
    
    def train_ae(self, dataloader, val_loader, eval_every, epochs):
        self.logger.info( 'pretraining AE' )
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for x in tqdm(dataloader, desc=f"[AE] epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)
                #loss = self.model.autoencoder.loss(x)
                loss = self.model.energy(x).mean()
                self.ae_opt.zero_grad()
                loss.backward()
                self.ae_opt.step()
                total_loss += loss.item() * x.size(0)
            self.logger.info(  f"[AE] epoch {epoch+1}: loss = {total_loss / len(dataloader.dataset):.10f}" )
            if (epoch + 1) % eval_every == 0:
                evaluate(self.model, val_loader, self.device, self.logger)

    def train_nae(self, dataloader, val_loader, eval_every, epochs):
        self.logger.info( 'training NAE' )
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            total_pos_energy = 0.0
            total_neg_energy = 0.0
            for x in tqdm(dataloader, desc=f"[NAE] epoch {epoch+1}/{epochs}"):
                x = x.to(self.device)
                pos_energy = self.model.energy(x).mean()
                neg_samples = self.model.sample_negative(x.size(0))
                neg_energy = self.model.energy(neg_samples).mean()
                loss = pos_energy - neg_energy
                self.nae_opt.zero_grad()
                loss.backward()
                self.nae_opt.step()
                total_loss += loss.item() * x.size(0)
                total_pos_energy += pos_energy.detach()
                total_neg_energy += neg_energy.detach()
            self.logger.info( f"[NAE] epoch {epoch+1}: loss = {total_loss / len(dataloader.dataset):.10f}, pos energy = {total_pos_energy / len(dataloader.dataset):.10f}, neg energy = {total_neg_energy / len(dataloader.dataset):.10f}" )
            if (epoch + 1) % eval_every == 0:
                evaluate(self.model, val_loader, self.device, self.logger)
