
import torch, torch.nn as nn
class VAE(nn.Module):
    def __init__(self, in_dim=16, latent_dim=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.mu = nn.Linear(32, latent_dim); self.logvar = nn.Linear(32, latent_dim)
        self.dec = nn.Sequential(nn.Linear(latent_dim,32), nn.ReLU(), nn.Linear(32,64), nn.ReLU(), nn.Linear(64,in_dim))
    def encode(self, x):
        h=self.enc(x); return self.mu(h), self.logvar(h)
    def reparam(self, mu, logvar):
        std=(0.5*logvar).exp(); eps=torch.randn_like(std); return mu+eps*std
    def decode(self, z): return self.dec(z)
    def forward(self, x):
        mu,lv=self.encode(x); z=self.reparam(mu,lv); xr=self.decode(z); return xr,mu,lv
def vae_loss(x,xr,mu,lv):
    rec=nn.functional.mse_loss(xr,x,reduction='mean')
    kld=-0.5*torch.mean(1+lv - mu.pow(2) - lv.exp())
    return rec + 1e-3*kld
