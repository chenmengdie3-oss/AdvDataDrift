
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def embed(real, fake, feats, seed=42):
    X = np.vstack([real[feats].values, fake[feats].values])
    y = np.array([0]*len(real) + [1]*len(fake))
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30.0, n_iter=1000, init="random")
    Z = tsne.fit_transform(X); return Z, y
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--real_dir", required=True); ap.add_argument("--models_dir", default="results"); ap.add_argument("--out", default="results/figs"); ap.add_argument("--seed", type=int, default=42); args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True); real=pd.read_csv(os.path.join(args.real_dir,"train.csv")); feats=[c for c in real.columns if c!='label']
    mapping={'VAE':'vae','GAN':'gan','CGAN':'cgan','WGAN':'wgan_gp','VAE-GAN':'vae_gan','VAE-CGAN':'vae_cgan'}
    plt.figure(figsize=(12,8)); i=1
    for title,sub in mapping.items():
        p=os.path.join(args.models_dir, sub, 'samples.csv'); 
        if not os.path.exists(p): continue
        fake=pd.read_csv(p); Z,y=embed(real,fake,feats,seed=args.seed)
        ax=plt.subplot(2,3,i); i+=1
        ax.scatter(Z[y==0,0],Z[y==0,1],s=6,alpha=0.6,label='Train (EX1-3)')
        ax.scatter(Z[y==1,0],Z[y==1,1],s=6,alpha=0.6,label=title)
        ax.set_title(title); ax.set_xticks([]); ax.set_yticks([])
        if i==2: ax.legend(loc='best', fontsize=8)
    plt.tight_layout(); outp=os.path.join(args.out,'tsne_grid.png'); plt.savefig(outp,dpi=200); plt.close(); print('Saved', outp)
if __name__=='__main__': main()
