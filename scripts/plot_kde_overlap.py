
import argparse, os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
def kde_overlap(ax, a, b, label_a='Real', label_b='Synthetic'):
    xmin, xmax = np.percentile(np.concatenate([a,b]), [1, 99]); xs = np.linspace(xmin, xmax, 256)
    ka, kb = gaussian_kde(a), gaussian_kde(b); ya, yb = ka(xs), kb(xs)
    ax.plot(xs, ya, label=label_a); ax.plot(xs, yb, label=label_b); ax.fill_between(xs, np.minimum(ya,yb), alpha=0.2)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--real_dir', required=True); ap.add_argument('--models_dir', default='results'); ap.add_argument('--out', default='results/figs/kde'); args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True); real=pd.read_csv(os.path.join(args.real_dir,"train.csv")); feats=[c for c in real.columns if c!='label']
    mapping={'GAN':'gan','CGAN':'cgan','WGAN':'wgan_gp','VAE-GAN':'vae_gan','VAE-CGAN':'vae_cgan','VAE':'vae'}
    for title,sub in mapping.items():
        path=os.path.join(args.models_dir, sub, 'samples.csv'); 
        if not os.path.exists(path): continue
        fake=pd.read_csv(path)
        fig,axs=plt.subplots(4,4,figsize=(12,10)); axs=axs.ravel()
        for i,f in enumerate(feats[:16]):
            kde_overlap(axs[i], real[f].values, fake[f].values, 'Train (EX1-3)', title); axs[i].set_title(f)
        plt.tight_layout(); outp=os.path.join(args.out, f'kde_overlap_{sub}.png'); plt.savefig(outp,dpi=200); plt.close(); print('Saved', outp)
if __name__=='__main__': main()
