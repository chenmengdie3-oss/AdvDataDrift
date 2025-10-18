import pathlib, pandas as pd

def test_data_exists():
    root = pathlib.Path("data")
    assert (root / "EX1.xlsx").exists(), "Missing EX1.xlsx example file"
