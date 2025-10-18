
import pandas as pd
import numpy as np
import os

LABEL_CANDIDATES = ["label","Label","y","Y","damage","Damage","tag","Tag"]

def read_excel_or_csv(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def find_label_column(df: pd.DataFrame):
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce")

def infer_label_if_missing(df: pd.DataFrame, n_feats: int = 16) -> pd.DataFrame:
    # Heuristic: derive a 3-class target from first n_feats columns by quantiles
    feats = [c for c in df.columns if c not in LABEL_CANDIDATES][:n_feats]
    Z = df[feats].copy().fillna(df[feats].median()).values
    vec = np.linspace(0.5, -0.2, min(len(feats), n_feats))
    z = (Z[:, :len(vec)] * vec[:Z.shape[1]]).sum(axis=1)
    q1, q2 = np.quantile(z, [0.33, 0.66])
    y = (z > q1).astype(int) + (z > q2).astype(int)
    df = df.copy()
    df["label"] = y
    return df
