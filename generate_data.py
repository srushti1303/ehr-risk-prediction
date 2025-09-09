# src/generate_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_patient(n_patients=5000):
    rows = []
    for pid in range(1, n_patients+1):
        age = np.random.randint(18, 90)
        sex = np.random.choice([0,1])  # 0 female, 1 male
        bmi = np.clip(np.random.normal(28, 6), 15, 50)
        # comorbidities: diabetes, hypertension, heart_disease (0/1)
        diabetes = np.random.binomial(1, 0.18 if age>50 else 0.08)
        htn = np.random.binomial(1, 0.3 if age>50 else 0.12)
        heart = np.random.binomial(1, 0.12 if age>60 else 0.03)
        # admission features
        length_of_stay = max(1, int(np.random.exponential(3)))
        num_prev_adm = np.random.poisson(1 if age>50 else 0.3)
        # labs (mean values)
        hgb = np.clip(np.random.normal(13.5 - 0.02*(age-50 if age>50 else 0), 1.2), 7, 18)
        glc = np.clip(np.random.normal(100 + 30*diabetes, 25), 60, 400)
        creat = np.clip(np.random.normal(1.0 + 0.01*(age-50 if age>50 else 0), 0.3), 0.3, 6)
        # risk logic for readmission (prob)
        base = -3.0
        base += 0.02*(age)
        base += 0.7*diabetes + 0.5*htn + 0.8*heart
        base += 0.04*(num_prev_adm)
        base += 0.05*(length_of_stay)
        base += 0.01*(glc-100)
        prob = 1/(1+np.exp(-base))
        readmit30 = np.random.binomial(1, prob)
        rows.append({
            "patient_id": f"P{pid:06d}",
            "age": age,
            "sex": sex,
            "bmi": round(bmi,1),
            "diabetes": int(diabetes),
            "hypertension": int(htn),
            "heart_disease": int(heart),
            "length_of_stay": length_of_stay,
            "num_prev_adm": int(num_prev_adm),
            "hgb": round(hgb,2),
            "glucose": round(glc,1),
            "creatinine": round(creat,2),
            "readmit_30d": int(readmit30)
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = generate_patient(8000)
    df.to_csv("data/synthetic_ehr.csv", index=False)
    print("Wrote ../data/synthetic_ehr.csv, shape:", df.shape)
