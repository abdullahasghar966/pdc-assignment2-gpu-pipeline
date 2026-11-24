"""Train a digit classifier on every CSV in data/ (old + new data together).

Uses the GPU (CUDA) when available, otherwise falls back to CPU.
Writes models/model.pt and models/metrics.json.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
MODEL_DIR = os.path.join(BASE, "models")
LABEL_COL = "label"


def load_all_data():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")))
    if not files:
        sys.exit("No CSV files found in data/")
    frames = [pd.read_csv(f) for f in files]
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    X = df.drop(columns=[LABEL_COL]).values.astype(np.float32) / 16.0
    y = df[LABEL_COL].values.astype(np.int64)
    print(f"Loaded {len(files)} file(s), {len(df)} unique rows")
    return X, y


def build_model():
    return nn.Sequential(
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 10),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    torch.manual_seed(42)

    X, y = load_all_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.tensor(X_tr, device=device)
    yt = torch.tensor(y_tr, device=device)

    epochs = 150
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        for i in range(0, len(Xt), 64):
            idx = perm[i:i + 64]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        if (epoch + 1) % 30 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_te, device=device)).argmax(dim=1).cpu().numpy()
    accuracy = float((preds == y_te).mean())
    print(f"Test accuracy: {accuracy:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.cpu().state_dict(), os.path.join(MODEL_DIR, "model.pt"))
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump({"accuracy": accuracy, "train_rows": int(len(X_tr)),
                   "test_rows": int(len(X_te)), "device": str(device)}, f, indent=2)


if __name__ == "__main__":
    main()
