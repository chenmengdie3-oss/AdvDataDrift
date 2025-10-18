
import pandas as pd

def split_train_valid(df: pd.DataFrame, valid_ratio=0.3, seed=42):
    # Stratified split by label
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=valid_ratio, random_state=seed, stratify=y)
    cols = df.drop(columns=["label"]).columns.tolist()
    import pandas as pd
    return (pd.DataFrame(Xtr, columns=cols).assign(label=ytr),
            pd.DataFrame(Xva, columns=cols).assign(label=yva))
