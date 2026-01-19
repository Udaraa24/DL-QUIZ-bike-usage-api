import os
import io
import zipfile
import requests
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from app.model_def import MLPRegressor

# We'll train on intuitive columns from the hourly dataset:
DATA_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

def load_hour_df() -> pd.DataFrame:
    r = requests.get(DATA_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("hour.csv") as f:
            return pd.read_csv(f)
            
# hr, temp, hum, windspeed, workingday, season, weathersit
FEATURES_NUM = ["hr", "temp", "hum", "windspeed"]
FEATURES_CAT = ["workingday", "season", "weathersit"]
TARGET = "cnt"  # total rental count

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def main():
    os.makedirs("model", exist_ok=True)

    df = load_hour_df()

    # Use only chosen columns
    df = df[FEATURES_NUM + FEATURES_CAT + [TARGET]].copy()

    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[TARGET].values.astype(np.float32)

    # Train/Val/Test split: 70/15/15
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42
    )

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURES_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
        ]
    )

    # Fit preprocess on TRAIN only (best practice)
    X_train_p = preprocess.fit_transform(X_train)
    X_val_p = preprocess.transform(X_val)
    X_test_p = preprocess.transform(X_test)

    # Convert to float32 dense (OneHotEncoder can be sparse)
    X_train_p = X_train_p.toarray().astype(np.float32) if hasattr(X_train_p, "toarray") else X_train_p.astype(np.float32)
    X_val_p = X_val_p.toarray().astype(np.float32) if hasattr(X_val_p, "toarray") else X_val_p.astype(np.float32)
    X_test_p = X_test_p.toarray().astype(np.float32) if hasattr(X_test_p, "toarray") else X_test_p.astype(np.float32)

    input_dim = X_train_p.shape[1]

    train_ds = TensorDataset(torch.from_numpy(X_train_p), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val_p), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test_p), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)

    device = torch.device("cpu")
    model = MLPRegressor(in_features=input_dim).to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, 26):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        preds = []
        trues = []
        with torch.no_grad():
            for xb, yb in val_loader:
                p = model(xb.to(device)).cpu().numpy()
                preds.append(p)
                trues.append(yb.numpy())

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)

        mae, rmse, r2 = metrics(trues, preds)
        print(f"Epoch {epoch:02d} | val MAE={mae:.2f} | val RMSE={rmse:.2f} | val R2={r2:.3f}")

        if rmse < best_val_rmse:
            best_val_rmse = rmse
            best_state = model.state_dict()

    model.load_state_dict(best_state)
    model.eval()

    # Final test metrics
    test_preds = []
    test_trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            p = model(xb.to(device)).cpu().numpy()
            test_preds.append(p)
            test_trues.append(yb.numpy())

    test_preds = np.concatenate(test_preds)
    test_trues = np.concatenate(test_trues)

    mae, rmse, r2 = metrics(test_trues, test_preds)
    print("\nTEST RESULTS")
    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.3f}")

    # Save artifacts
    torch.save({"model_state_dict": model.state_dict(), "input_dim": input_dim}, "model/model.pt")
    joblib.dump(preprocess, "model/preprocess.pkl")
    print("\nSaved model/model.pt and model/preprocess.pkl")

if __name__ == "__main__":
    main()
