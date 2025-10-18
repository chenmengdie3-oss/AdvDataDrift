
import argparse, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--metrics_dir', required=True); ap.add_argument('--out', default='results/figs/boxplots'); args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    files=[f for f in os.listdir(args.metrics_dir) if f.endswith('_metrics.csv')]
    data=[]
    for f in files:
        df=pd.read_csv(os.path.join(args.metrics_dir,f)); model=f.split('_metrics.csv')[0]
        data.append((model, df['KS'].values, df['Wasserstein'].values))
    if not data: 
        print('No metrics found.'); return
    plt.figure(figsize=(8,4)); plt.boxplot([d[1] for d in data], labels=[d[0] for d in data])
    plt.title('KS distribution across features'); plt.ylabel('KS'); plt.tight_layout()
    out1=os.path.join(args.out,'ks_boxplot.png'); plt.savefig(out1,dpi=200); plt.close(); print('Saved', out1)
    plt.figure(figsize=(8,4)); plt.boxplot([d[2] for d in data], labels=[d[0] for d in data])
    plt.title('Wasserstein distribution across features'); plt.ylabel('Wasserstein'); plt.tight_layout()
    out2=os.path.join(args.out,'wasserstein_boxplot.png'); plt.savefig(out2,dpi=200); plt.close(); print('Saved', out2)
if __name__=='__main__': main()
