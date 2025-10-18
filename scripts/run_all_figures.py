#!/usr/bin/env python
import argparse, subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run(cmd):
    print(f"[run] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[warn] command failed with code {r.returncode}: {' '.join(cmd)}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='data/processed', help='Processed data directory')
    ap.add_argument('--out', default='results/figures', help='Output directory for figures')
    args = ap.parse_args()

    (ROOT / args.out).mkdir(parents=True, exist_ok=True)

    cmds = [
        [sys.executable, 'scripts/plot_tsne_grid.py', '--data_dir', args.data_dir, '--out', args.out],
        [sys.executable, 'scripts/plot_kde_overlap.py', '--data_dir', args.data_dir, '--out', args.out],
        [sys.executable, 'scripts/plot_boxplots.py', '--data_dir', args.data_dir, '--out', args.out],
        [sys.executable, 'scripts/plot_metrics.py', '--data_dir', args.data_dir, '--out', args.out],
    ]
    for c in cmds:
        run(c)

if __name__ == '__main__':
    main()
