
import argparse, os, torch, numpy as np, pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from .models_vae import VAE, vae_loss

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--latent", type=int, default=8)
    ap.add_argument("--out", default="results/vae")
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.data)
    X = torch.tensor(df.drop(columns=["label"]).values, dtype=torch.float32)
    cols = df.drop(columns=["label"]).columns.tolist()

    dl = DataLoader(TensorDataset(X), batch_size=args.batch, shuffle=True)
    model = VAE(in_dim=X.shape[1], latent_dim=args.latent)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for ep in range(args.epochs):
        losses=[]
        for (xb,) in dl:
            xr,mu,lv = model(xb)
            loss = vae_loss(xb,xr,mu,lv)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[VAE] epoch {ep+1}/{args.epochs} loss={np.mean(losses):.4f}")

    with torch.no_grad():
        z = torch.randn(500, args.latent)
        synth = model.decode(z).cpu().numpy()
    pd.DataFrame(synth, columns=cols).to_csv(os.path.join(args.out,"samples.csv"), index=False)

if __name__=="__main__": main()
