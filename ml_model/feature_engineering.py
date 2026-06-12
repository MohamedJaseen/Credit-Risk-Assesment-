"""
ml_model/feature_engineering.py
================================
Converts raw user inputs into the 13-feature vector the ML models expect.
"""

import numpy as np
import pickle
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

FEATURE_NAMES = [
    "age", "monthly_income", "loan_amount", "loan_tenure",
    "existing_emis", "missed_payments", "credit_utilization",
    "credit_history_years", "dti", "total_dti",
    "income_loan_ratio", "employment_encoded", "emi_new",
]

EMPLOYMENT_MAP = {"salaried": 0, "self-employed": 1, "unemployed": 2}
MONTHLY_RATE   = 0.01   # 12% p.a. → 1% per month


def compute_features(raw: dict) -> dict:
    """
    Derive all engineered features from raw user inputs.

    Parameters
    ----------
    raw : dict with keys matching the loan application form fields

    Returns
    -------
    dict of feature_name → float
    """
    income  = max(float(raw["monthly_income"]), 1.0)
    loan    = float(raw["loan_amount"])
    tenure  = max(int(raw["loan_tenure"]), 1)
    emis    = float(raw["existing_emis"])
    emp_str = str(raw["employment_type"]).lower()

    # Derived features
    dti         = round(min(emis / income, 2.0), 4)
    emi_new     = loan * MONTHLY_RATE / (1 - (1 + MONTHLY_RATE) ** -tenure)
    total_dti   = round(min((emis + emi_new) / income, 2.0), 4)
    inc_loan_r  = round(min(income / max(loan, 1), 5.0), 4)
    emp_enc     = float(EMPLOYMENT_MAP.get(emp_str, 1))

    return {
        "age":                  float(raw["age"]),
        "monthly_income":       income,
        "loan_amount":          loan,
        "loan_tenure":          float(tenure),
        "existing_emis":        emis,
        "missed_payments":      float(raw["missed_payments"]),
        "credit_utilization":   float(raw["credit_utilization"]),
        "credit_history_years": float(raw["credit_history_years"]),
        "dti":                  dti,
        "total_dti":            total_dti,
        "income_loan_ratio":    inc_loan_r,
        "employment_encoded":   emp_enc,
        "emi_new":              round(emi_new, 2),
    }


def to_array(feats: dict) -> np.ndarray:
    """Convert feature dict to ordered numpy array."""
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)


def scale(arr: np.ndarray) -> np.ndarray:
    """Apply saved StandardScaler. Returns raw array if scaler not found."""
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            sc = pickle.load(f)
        return sc.transform(arr.reshape(1, -1))[0]
    return arr
