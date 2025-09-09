# src/api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="EHR Readmission Risk API")

MODEL_PATH = "models/risk_model.json"
SCALER_PATH = "models/scaler.npz"

# Load XGBoost model via native API
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Reconstruct StandardScaler from saved parameters
scaler_np = np.load(SCALER_PATH)
scaler = StandardScaler()

scaler.mean_ = scaler_np["mean"]
scaler.scale_ = scaler_np["scale"]
scaler.var_ = scaler_np["var"] if "var" in scaler_np.files else np.square(scaler.scale_)
scaler.n_features_in_ = scaler.mean_.shape[0]
# set n_samples_seen_ if available (some sklearn versions use it)
try:
    scaler.n_samples_seen_ = int(np.asarray(scaler_np["n_samples_seen"]))
except Exception:
    scaler.n_samples_seen_ = getattr(scaler, "n_samples_seen_", 0)

FEATURES = ['age','sex','bmi','length_of_stay','num_prev_adm','hgb','glucose','creatinine','comorbidity_score']

class Patient(BaseModel):
    age: int
    sex: int
    bmi: float
    length_of_stay: int
    num_prev_adm: int
    hgb: float
    glucose: float
    creatinine: float
    comorbidity_score: int = None
    diabetes: int = None
    hypertension: int = None
    heart_disease: int = None

@app.post("/predict")
def predict_readmission(p: Patient):
    # compute comorbidity if not provided
    comorb = p.comorbidity_score
    if comorb is None:
        comorb = sum([int(x) for x in [p.diabetes or 0, p.hypertension or 0, p.heart_disease or 0]])
    x = np.array([[p.age, p.sex, p.bmi, p.length_of_stay, p.num_prev_adm, p.hgb, p.glucose, p.creatinine, comorb]])
    x_scaled = scaler.transform(x)
    proba = float(model.predict_proba(x_scaled)[:,1][0])
    return {"risk_score": round(proba,4), "high_risk": proba > 0.5}

@app.get("/health")
def health():
    return {"status":"ok"}
