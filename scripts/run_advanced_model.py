
import argparse, os, pandas as pd, glob
from src.evaluate_rf import train_eval_rf, save_confusion

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--models_dir", default="results")
    ap.add_argument("--out", default="results/advanced")
    args=ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tr = pd.read_csv(os.path.join(args.data_dir,"train.csv"))
    te = pd.read_csv(os.path.join(args.data_dir,"test_ex4.csv"))

    # Augment using any samples.csv found under models_dir
    aug = []
    for p in glob.glob(os.path.join(args.models_dir,"**","samples.csv"), recursive=True):
        df = pd.read_csv(p)
        if "label" not in df.columns:
            # assign labels by nearest class proportion from train
            major = tr["label"].mode().iloc[0]
            df["label"] = major
        aug.append(df)
    if aug:
        import pandas as pd
        aug_df = pd.concat(aug, axis=0, ignore_index=True).iloc[:1500]  # cap to control runtime
        tr_aug = pd.concat([tr, aug_df], axis=0, ignore_index=True)
    else:
        tr_aug = tr

    acc, cm, _ = train_eval_rf(tr_aug, te, trees=10000)
    save_confusion(cm, acc, os.path.join(args.out, "confusion_rf_advanced.png"), title="Advanced RF (10k trees)")
    with open(os.path.join(args.out,"metrics.txt"),"w") as f:
        f.write(f"Advanced Accuracy (EX4 test): {acc}\nCM:\n{cm}\n")
    print("Done.")

if __name__=="__main__": main()
