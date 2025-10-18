
import os
import pandas as pd
import numpy as np
from .io_utils import read_excel_or_csv, find_label_column, ensure_numeric, infer_label_if_missing
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE

def load_and_prepare(paths, use_smote=True, random_state=42):
    # Read datasets
    dfs = []
    for p in paths:
        df = read_excel_or_csv(p)
        df = ensure_numeric(df)
        lab = find_label_column(df)
        if lab is None:
            df = infer_label_if_missing(df)
            lab = "label"
        # Keep first 16 feature columns if there are more
        feats = [c for c in df.columns if c != lab][:16]
        df = df[feats + [lab]].dropna()
        dfs.append(df)

    # MinMax fit on EX1+EX2+EX3 (train side) to avoid leakage from EX4
    scaler = MinMaxScaler()
    train_df = pd.concat(dfs[:3], axis=0, ignore_index=True)
    test_df  = dfs[3]

    Xtr = train_df.drop(columns=["label"]).values
    Xte = test_df.drop(columns=["label"]).values
    scaler.fit(Xtr)
    Xtr = scaler.transform(Xtr); Xte = scaler.transform(Xte)

    train_df = pd.DataFrame(Xtr, columns=[f"ln(X{i})" for i in range(1,17)]).assign(label=train_df["label"].values)
    test_df  = pd.DataFrame(Xte, columns=[f"ln(X{i})" for i in range(1,17)]).assign(label=test_df["label"].values)

    if use_smote:
        sm = SMOTE(random_state=random_state)
        Xs, ys = sm.fit_resample(train_df.drop(columns=["label"]).values, train_df["label"].values)
        train_df = pd.DataFrame(Xs, columns=[f"ln(X{i})" for i in range(1,17)]).assign(label=ys)

    return train_df, test_df
