
import argparse, os, torch, numpy as np, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from .models_gan import MLP_G, MLP_D

def gradient_penalty(D, real, fake):
    b = real.size(0); eps = torch.rand(b,1, device=real.device).expand_as(real)
    inter = eps*real + (1-eps)*fake; inter.requires_grad_(True)
    d = D(inter); g = torch.autograd.grad(d, inter, torch.ones_like(d), create_graph=True, retain_graph=True)[0]
    return ((g.view(b,-1).norm(2, dim=1)-1)**2).mean()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--gan_type", choices=["gan","cgan","wgan-gp"], default="wgan-gp")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--z", type=int, default=32)
    ap.add_argument("--out", default="results/wgan_gp")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.data)
    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    cols = df.drop(columns=["label"]).columns.tolist()
    n_classes = int(y.max().item())+1

    dl = DataLoader(TensorDataset(X,y), batch_size=args.batch, shuffle=True)
    cond = (args.gan_type=="cgan")
    G = MLP_G(args.z, X.shape[1], n_classes=(n_classes if cond else None))
    D = MLP_D(X.shape[1], n_classes=(n_classes if cond else None))

    if args.gan_type in ["gan","cgan"]:
        optG = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5,0.9))
        optD = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5,0.9))
        bce = torch.nn.BCEWithLogitsLoss()
    else:
        optG = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0,0.9))
        optD = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0,0.9))

    for ep in range(args.epochs):
        for xb,yb in dl:
            z = torch.randn(xb.size(0), args.z)
            if cond:
                y1 = torch.nn.functional.one_hot(yb, num_classes=n_classes).float()
                fake = G(z, y1); d_real = D(xb, y1); d_fake = D(fake.detach(), y1)
            else:
                fake = G(z); d_real = D(xb); d_fake = D(fake.detach())

            if args.gan_type in ["gan","cgan"]:
                lossD = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
                optD.zero_grad(); lossD.backward(); optD.step()
                d_fake2 = D(G(z, y1), y1) if cond else D(G(z))
                lossG = bce(d_fake2, torch.ones_like(d_fake2))
                optG.zero_grad(); lossG.backward(); optG.step()
            else:
                gp = gradient_penalty(lambda t: D(t), xb, fake)
                lossD = d_fake.mean() - d_real.mean() + 10.0*gp
                optD.zero_grad(); lossD.backward(); optD.step()
                fake = G(z); lossG = - D(fake).mean()
                optG.zero_grad(); lossG.backward(); optG.step()
        print(f"[{args.gan_type}] epoch {ep+1}/{args.epochs}")

    with torch.no_grad():
        z = torch.randn(500, args.z)
        if cond:
            labels = torch.arange(n_classes).repeat_interleave(500//n_classes + 1)[:500]
            onehot = torch.nn.functional.one_hot(labels, num_classes=n_classes).float()
            synth = G(z, onehot).cpu().numpy()
            df_out = pd.DataFrame(synth, columns=cols); df_out["label"]=labels.numpy()
        else:
            synth = G(z).cpu().numpy()
            df_out = pd.DataFrame(synth, columns=cols)
    df_out.to_csv(os.path.join(args.out, "samples.csv"), index=False)

if __name__=="__main__": main()
