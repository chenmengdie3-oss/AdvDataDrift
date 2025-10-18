
import torch, torch.nn as nn
class VAE_Enc(nn.Module):
    def __init__(self, in_dim=16, latent_dim=8):
        super().__init__(); 
        self.net = nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,32), nn.ReLU())
        self.mu = nn.Linear(32, latent_dim); self.logvar = nn.Linear(32, latent_dim)
    def forward(self, x):
        h=self.net(x); return self.mu(h), self.logvar(h)
class VAE_Dec(nn.Module):
    def __init__(self, out_dim=16, latent_dim=8, n_classes=None):
        super().__init__(); self.n_classes=n_classes
        in_dim = latent_dim + (0 if n_classes is None else n_classes)
        self.net = nn.Sequential(nn.Linear(in_dim,32), nn.ReLU(), nn.Linear(32,64), nn.ReLU(), nn.Linear(64,out_dim))
    def forward(self, z, y_onehot=None):
        if self.n_classes is not None and y_onehot is not None:
            z = torch.cat([z,y_onehot], dim=1)
        return self.net(z)
class Hybrid_D(nn.Module):
    def __init__(self, in_dim=16, n_classes=None):
        super().__init__(); self.n_classes=n_classes
        in_dim2 = in_dim + (0 if n_classes is None else n_classes)
        self.net = nn.Sequential(nn.Linear(in_dim2,64), nn.LeakyReLU(0.2), nn.Linear(64,64), nn.LeakyReLU(0.2), nn.Linear(64,1))
    def forward(self, x, y_onehot=None):
        if self.n_classes is not None and y_onehot is not None:
            x = torch.cat([x,y_onehot], dim=1)
        return self.net(x)
def kld(mu, logvar): return -0.5*torch.mean(1+logvar - mu.pow(2) - logvar.exp())
def recon(x,xrec): return nn.functional.mse_loss(xrec,x,reduction='mean')
