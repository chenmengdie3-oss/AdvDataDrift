
import argparse, os
from src.preprocess import load_and_prepare

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--smote", action="store_true")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--ex1", default="data/EX1.xlsx")
    ap.add_argument("--ex2", default="data/EX2-1.xlsx")
    ap.add_argument("--ex3", default="data/EX3-1.xlsx")
    ap.add_argument("--ex4", default="data/EX4.xlsx")
    args=ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_df, test_df = load_and_prepare([args.ex1,args.ex2,args.ex3,args.ex4], use_smote=args.smote)
    train_df.to_csv(os.path.join(args.out_dir,"train.csv"), index=False)
    test_df.to_csv(os.path.join(args.out_dir,"test_ex4.csv"), index=False)
    print("Saved:", os.path.join(args.out_dir,"train.csv"), os.path.join(args.out_dir,"test_ex4.csv"))

if __name__=="__main__": main()
