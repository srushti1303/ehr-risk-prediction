# src/train_model.py
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
import os
import platform
import xgboost as xgb
import sklearn

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def train():
    X_train = pd.read_csv(f"{MODELS_DIR}/X_train.csv")
    X_test = pd.read_csv(f"{MODELS_DIR}/X_test.csv")
    y_train = pd.read_csv(f"{MODELS_DIR}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{MODELS_DIR}/y_test.csv").squeeze()

    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
    model.fit(X_train, y_train)

    # predict and evaluate
    pred_proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, pred_proba)
    auprc = average_precision_score(y_test, pred_proba)
    brier = brier_score_loss(y_test, pred_proba)
    print(f"AUC: {auc:.4f}, AUPRC: {auprc:.4f}, Brier: {brier:.4f}")

    # Save XGBoost model in native JSON format (version-safe)
    model.save_model(f"{MODELS_DIR}/risk_model.json")
    print("Saved xgboost model to models/risk_model.json")
    
    # Save a tiny metrics file
    with open(f"{MODELS_DIR}/metrics.txt","w") as f:
        f.write(f"AUC: {auc:.4f}\nAUPRC: {auprc:.4f}\nBrier: {brier:.4f}\n")

if __name__ == "__main__":
    train()
