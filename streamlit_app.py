# dashboard/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "models/risk_model.json"
SCALER_PATH = "models/scaler.npz"
SHAP_FEATURES_PATH = "models/shap_feature_importance.csv"

# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Load scaler reconstructed from npz
scaler_np = np.load(SCALER_PATH)
scaler = StandardScaler()
scaler.mean_ = scaler_np["mean"]
scaler.scale_ = scaler_np["scale"]
scaler.var_ = scaler_np["var"] if "var" in scaler_np.files else np.square(scaler.scale_)
scaler.n_features_in_ = scaler.mean_.shape[0]
try:
    scaler.n_samples_seen_ = int(np.asarray(scaler_np["n_samples_seen"]))
except Exception:
    scaler.n_samples_seen_ = getattr(scaler, "n_samples_seen_", 0)

st.title("EHR Readmission Risk Demo")

st.markdown("Upload a CSV with columns: age, sex, bmi, length_of_stay, num_prev_adm, hgb, glucose, creatinine, comorbidity_score")

uploaded = st.file_uploader("CSV", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    df_scaled = scaler.transform(df)
    probs = model.predict_proba(df_scaled)[:,1]
    df['risk_score'] = probs
    st.dataframe(df.head(20))
    st.bar_chart(df['risk_score'].value_counts(bins=10))

st.sidebar.header("Single patient input")
age = st.sidebar.slider("Age", 18, 95, 65)
sex = st.sidebar.selectbox("Sex", [0,1])
bmi = st.sidebar.slider("BMI", 15.0, 50.0, 28.0)
los = st.sidebar.slider("Length of stay", 1, 30, 3)
npa = st.sidebar.slider("Num prev admissions", 0, 10, 1)
hgb = st.sidebar.slider("Hgb", 7.0, 18.0, 13.5)
glc = st.sidebar.slider("Glucose", 60.0, 400.0, 110.0)
creat = st.sidebar.slider("Creatinine", 0.3, 6.0, 1.0)
comorb = st.sidebar.slider("Comorbidity score", 0, 3, 1)

if st.sidebar.button("Predict"):
    x = np.array([[age, sex, bmi, los, npa, hgb, glc, creat, comorb]])
    x_sc = scaler.transform(x)
    proba = float(model.predict_proba(x_sc)[:,1][0])
    st.write("Risk score:", round(proba,4))
    if os.path.exists(SHAP_FEATURES_PATH):
        st.write("Top features (precomputed SHAP):")
        st.table(pd.read_csv(SHAP_FEATURES_PATH).head(10))
    else:
        st.write("(Run src/explain.py to compute SHAP feature importance.)")
