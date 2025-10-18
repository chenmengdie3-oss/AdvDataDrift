
import torch, torch.nn as nn
class MLP_G(nn.Module):
    def __init__(self, z_dim=32, out_dim=16, n_classes=None):
        super().__init__(); self.n_classes=n_classes
        in_dim = z_dim + (0 if n_classes is None else n_classes)
        self.net = nn.Sequential(nn.Linear(in_dim,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU(), nn.Linear(64,out_dim))
    def forward(self, z, y_onehot=None):
        if self.n_classes is not None and y_onehot is not None:
            z = torch.cat([z,y_onehot], dim=1)
        return self.net(z)
class MLP_D(nn.Module):
    def __init__(self, in_dim=16, n_classes=None):
        super().__init__(); self.n_classes=n_classes
        in_dim2 = in_dim + (0 if n_classes is None else n_classes)
        self.net = nn.Sequential(nn.Linear(in_dim2,64), nn.LeakyReLU(0.2), nn.Linear(64,64), nn.LeakyReLU(0.2), nn.Linear(64,1))
    def forward(self, x, y_onehot=None):
        if self.n_classes is not None and y_onehot is not None:
            x = torch.cat([x,y_onehot], dim=1)
        return self.net(x)
