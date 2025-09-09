# src/explain.py
import joblib
import pandas as pd
import shap
import os
import xgboost as xgb
import numpy as np

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def explain_top():
    model = xgb.XGBClassifier()
    model.load_model(f"{MODELS_DIR}/risk_model.json")
    X_test = pd.read_csv(f"{MODELS_DIR}/X_test.csv")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # save shap summary as a dataframe (mean abs shap)
    mean_abs = pd.DataFrame({
        "feature": X_test.columns,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    mean_abs.to_csv(f"{MODELS_DIR}/shap_feature_importance.csv", index=False)
    print("Saved shap_feature_importance.csv")

if __name__ == "__main__":
    explain_top()
