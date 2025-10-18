
import argparse, os, torch, numpy as np, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from .models_hybrid import VAE_Enc, VAE_Dec, Hybrid_D, kld, recon

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--hybrid", choices=["vae-gan","vae-cgan"], default="vae-gan")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--out", default="results/vae_gan")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    cols = df.drop(columns=["label"]).columns.tolist()
    n_classes = int(y.max().item())+1

    cond = (args.hybrid=="vae-cgan")
    E = VAE_Enc(X.shape[1], args.latent)
    G = VAE_Dec(X.shape[1], args.latent, n_classes=(n_classes if cond else None))
    D = Hybrid_D(X.shape[1], n_classes=(n_classes if cond else None))

    optE = torch.optim.Adam(E.parameters(), lr=1e-3)
    optG = torch.optim.Adam(G.parameters(), lr=1e-3)
    optD = torch.optim.Adam(D.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss()

    dl = DataLoader(TensorDataset(X,y), batch_size=args.batch, shuffle=True)

    for ep in range(args.epochs):
        for xb,yb in dl:
            mu, logvar = E(xb); std=(0.5*logvar).exp(); z = mu + torch.randn_like(std)*std
            if cond:
                y1 = torch.nn.functional.one_hot(yb, num_classes=n_classes).float()
                xrec = G(z, y1); d_real = D(xb, y1); d_fake = D(xrec.detach(), y1)
            else:
                xrec = G(z); d_real = D(xb); d_fake = D(xrec.detach())
            lossD = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            optD.zero_grad(); lossD.backward(); optD.step()
            d_fake2 = D(xrec, y1) if cond else D(xrec); adv = bce(d_fake2, torch.ones_like(d_fake2))
            lossGE = recon(xb,xrec) + 1e-3*kld(mu,logvar) + 1e-2*adv
            optE.zero_grad(); optG.zero_grad(); lossGE.backward(); optE.step(); optG.step()
        print(f"[{args.hybrid}] epoch {ep+1}/{args.epochs}")

    with torch.no_grad():
        z = torch.randn(500, args.latent)
        if cond:
            labels = torch.arange(n_classes).repeat_interleave(500//n_classes + 1)[:500]
            onehot = torch.nn.functional.one_hot(labels, num_classes=n_classes).float()
            synth = G(z, onehot).cpu().numpy(); df_out = pd.DataFrame(synth, columns=cols); df_out["label"]=labels.numpy()
        else:
            synth = G(z).cpu().numpy(); df_out = pd.DataFrame(synth, columns=cols)
    df_out.to_csv(os.path.join(args.out,"samples.csv"), index=False)

if __name__=="__main__": main()
