
import argparse, os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def train_eval_rf(train_df, test_df, trees=1000, seed=42):
    Xtr = train_df.drop(columns=["label"]).values; ytr = train_df["label"].values
    Xte = test_df.drop(columns=["label"]).values;  yte = test_df["label"].values
    rf = RandomForestClassifier(n_estimators=trees, random_state=seed, n_jobs=-1)
    rf.fit(Xtr,ytr); yp = rf.predict(Xte); acc = accuracy_score(yte, yp)
    cm = confusion_matrix(yte, yp)
    return acc, cm, rf

def save_confusion(cm, acc, out_path, title="Confusion"):
    fig, ax = plt.subplots(figsize=(4,4))
    disp = ConfusionMatrixDisplay(cm); disp.plot(ax=ax, colorbar=False)
    plt.title(f"{title} (acc={acc:.3f})"); plt.tight_layout()
    plt.savefig(out_path, dpi=200); plt.close()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--trees", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args=ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tr = pd.read_csv(args.train); te = pd.read_csv(args.test)

    acc, cm, _ = train_eval_rf(tr, te, trees=args.trees)
    save_confusion(cm, acc, os.path.join(args.out, "confusion_rf.png"), title="RF")
    with open(os.path.join(args.out,"metrics.txt"),"w") as f:
        f.write(f"Accuracy: {acc}\nCM:\n{cm}\n")

if __name__=="__main__": main()
