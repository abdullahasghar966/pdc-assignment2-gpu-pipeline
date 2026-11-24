"""Create data/digits_batch1.csv from scikit-learn's handwritten digits.

Run once. Later batches can be added as more CSV files in data/ with the
same columns (f0..f63 pixel values 0-16, label 0-9); train.py merges them.
"""
import os

import pandas as pd
from sklearn.datasets import load_digits

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "data", "digits_batch1.csv")

digits = load_digits()
df = pd.DataFrame(digits.data, columns=[f"f{i}" for i in range(64)])
df["label"] = digits.target
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {len(df)} rows to {OUT}")
