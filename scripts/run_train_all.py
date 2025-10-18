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
    ap.add_argument('--train_csv', default='data/processed/train.csv')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    # Prefer Python trainers if available; otherwise fall back to shell script
    if (ROOT / 'src' / 'train_vae.py').exists():
        run([sys.executable, 'src/train_vae.py', '--data', args.train_csv])
    if (ROOT / 'src' / 'train_gan.py').exists():
        run([sys.executable, 'src/train_gan.py', '--data', args.train_csv])
    if (ROOT / 'src' / 'train_hybrid.py').exists():
        run([sys.executable, 'src/train_hybrid.py', '--data', args.train_csv])

    # Fallback to legacy shell script if present
    if (ROOT / 'scripts' / 'run_train_all.sh').exists():
        run(['bash', 'scripts/run_train_all.sh', args.train_csv, str(args.workers)])

if __name__ == '__main__':
    main()
