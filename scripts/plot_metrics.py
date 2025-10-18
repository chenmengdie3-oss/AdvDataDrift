
import argparse, os, pandas as pd, numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
def compute(real_df, fake_df):
    feats=[c for c in real_df.columns if c!='label' and c in fake_df.columns]
    rows=[]
    for c in feats:
        r,f=real_df[c].values, fake_df[c].values
        ks=ks_2samp(r,f).statistic; w=wasserstein_distance(r,f)
        rows.append({'feature':c,'KS':ks,'Wasserstein':w})
    return pd.DataFrame(rows)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--real_dir', required=True); ap.add_argument('--models_dir', default='results'); ap.add_argument('--out', default='results/metrics'); args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True); real=pd.read_csv(os.path.join(args.real_dir,"train.csv"))
    models={'vae':'VAE','gan':'GAN','cgan':'CGAN','wgan_gp':'WGAN','vae_gan':'VAE-GAN','vae_cgan':'VAE-CGAN'}
    summary=[]
    for sub,name in models.items():
        path=os.path.join(args.models_dir, sub, 'samples.csv')
        if os.path.exists(path):
            fake=pd.read_csv(path); df=compute(real,fake); df.to_csv(os.path.join(args.out, f'{sub}_metrics.csv'), index=False)
            summary.append((name, df['KS'].mean(), df['Wasserstein'].mean()))
    if summary:
        sm=pd.DataFrame(summary, columns=['model','KS_mean','Wasserstein_mean']).sort_values('KS_mean')
        sm.to_csv(os.path.join(args.out,'summary.csv'), index=False); print(sm)
if __name__=='__main__': main()
