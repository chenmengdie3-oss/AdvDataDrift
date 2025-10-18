#!/usr/bin/env python
import argparse, subprocess, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent

def run(cmd):
    print(f"[run] {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"[warn] command failed with code {r.returncode}: {' '.join(cmd)}", file=sys.stderr)
    return r.returncode

def stage_preprocess():
    return run([sys.executable, 'scripts/preprocess_data.py', '--smote', '--out_dir', 'data/processed'])

def stage_baseline():
    return run([sys.executable, 'scripts/run_baseline.py', '--data_dir', 'data/processed', '--out', 'results/baseline'])

def stage_train():
    # Try python runner first, fallback to shell
    if (ROOT / 'scripts' / 'run_train_all.py').exists():
        return run([sys.executable, 'scripts/run_train_all.py', '--train_csv', 'data/processed/train.csv', '--workers', '8'])
    return run(['bash', 'scripts/run_train_all.sh', 'data/processed/train.csv', '8'])

def stage_figures():
    # Prefer python runner
    if (ROOT / 'scripts' / 'run_all_figures.py').exists():
        return run([sys.executable, 'scripts/run_all_figures.py', '--data_dir', 'data/processed', '--out', 'results/figures'])
    return run(['bash', 'scripts/run_all_figures.sh', 'data/processed'])

def main():
    ap = argparse.ArgumentParser(description="Unified entry for the corrosion data-drift pipeline")
    ap.add_argument('--stage', choices=['full', 'preprocess', 'baseline', 'train', 'figures'], default='full')
    args = ap.parse_args()

    if args.stage == 'preprocess':
        sys.exit(stage_preprocess())
    if args.stage == 'baseline':
        sys.exit(stage_baseline())
    if args.stage == 'train':
        sys.exit(stage_train())
    if args.stage == 'figures':
        sys.exit(stage_figures())

    # Full pipeline
    rc = stage_preprocess();  assert rc == 0 or rc is None
    rc = stage_baseline();    assert rc == 0 or rc is None
    rc = stage_train();       assert rc == 0 or rc is None
    rc = stage_figures();     assert rc == 0 or rc is None
    print("[done] Full pipeline finished. See results/ for outputs.")

if __name__ == '__main__':
    main()
