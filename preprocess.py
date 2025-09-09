# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

DATA_PATH = "data/synthetic_ehr.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def preprocess(df):
    # Create a few derived features
    df = df.copy()
    df['age_binned'] = pd.cut(df['age'], bins=[17,30,45,60,75,100], labels=False)
    df['comorbidity_score'] = df[['diabetes','hypertension','heart_disease']].sum(axis=1)
    # features and label
    features = ['age','sex','bmi','length_of_stay','num_prev_adm','hgb','glucose','creatinine','comorbidity_score']
    X = df[features]
    y = df['readmit_30d']
    # train/test split (stratify)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    # Save scaler and datasets
    np.savez(
        f"{OUT_DIR}/scaler.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=getattr(scaler, "var_", np.square(scaler.scale_)),
        n_samples_seen=getattr(scaler, "n_samples_seen_", 0)
    )
    pd.DataFrame(X_train_sc, columns=features).to_csv(f"{OUT_DIR}/X_train.csv", index=False)
    pd.DataFrame(X_test_sc, columns=features).to_csv(f"{OUT_DIR}/X_test.csv", index=False)
    y_train.to_csv(f"{OUT_DIR}/y_train.csv", index=False)
    y_test.to_csv(f"{OUT_DIR}/y_test.csv", index=False)
    print("Preprocessing done. Saved in models/")
    return features

if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    preprocess(df)
