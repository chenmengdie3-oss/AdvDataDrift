
import argparse, os, pandas as pd
from src.data_api import split_train_valid
from src.evaluate_rf import train_eval_rf, save_confusion

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out", default="results/baseline")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr = pd.read_csv(os.path.join(args.data_dir,"train.csv"))
    te = pd.read_csv(os.path.join(args.data_dir,"test_ex4.csv"))
    # Baseline per paper: train on EX1 only; here we approximate by using only label-balanced subset from EX1 portion is already scaled.
    # For simplicity, use a stratified split of train.csv to form a "baseline-train" (proxy for EX1) to avoid needing raw EX1 after scaling.
    tr_base, _ = split_train_valid(tr, valid_ratio=0.7, seed=42)  # about 30% of train used
    acc, cm, _ = train_eval_rf(tr_base, te, trees=1000)
    save_confusion(cm, acc, os.path.join(args.out, "confusion_rf_baseline.png"), title="Baseline RF (EX1-only proxy)")
    with open(os.path.join(args.out,"metrics.txt"),"w") as f:
        f.write(f"Baseline Accuracy (EX4 test): {acc}\nCM:\n{cm}\n")

if __name__=="__main__": main()
