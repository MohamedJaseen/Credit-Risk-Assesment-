"""
ml_model/explain.py
====================
SHAP-based explainability — fast approximation for real-time use.
"""

import numpy as np
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FEATURE_NAMES = [
    "age", "monthly_income", "loan_amount", "loan_tenure",
    "existing_emis", "missed_payments", "credit_utilization",
    "credit_history_years", "dti", "total_dti",
    "income_loan_ratio", "employment_encoded", "emi_new",
]

FEATURE_LABELS = {
    "age":                   "Applicant Age",
    "monthly_income":        "Monthly Income",
    "loan_amount":           "Loan Amount",
    "loan_tenure":           "Loan Tenure",
    "existing_emis":         "Existing EMIs",
    "missed_payments":       "Missed Payments",
    "credit_utilization":    "Credit Utilization %",
    "credit_history_years":  "Credit History (yrs)",
    "dti":                   "Debt-to-Income Ratio",
    "total_dti":             "Total DTI (w/ new loan)",
    "income_loan_ratio":     "Income-to-Loan Ratio",
    "employment_encoded":    "Employment Type",
    "emi_new":               "New EMI Estimate",
}

# Per-feature risk sensitivity (domain knowledge)
_WEIGHTS = np.array([
    0.25, 0.80, 1.10, 0.40, 0.60,
    1.60, 1.30, 0.75, 1.00, 1.20,
    0.90, 0.50, 0.45,
])


def approx_shap(scaled: np.ndarray, prob: float) -> np.ndarray:
    """
    Fast SHAP proxy: deviation × risk weight × confidence factor.
    Good enough for a college-level explanation.
    """
    base     = np.zeros_like(scaled)
    dev      = scaled - base
    weight   = (prob - 0.5) * 2.0
    return dev * weight * _WEIGHTS


def shap_to_text(shap_vals: np.ndarray, prob: float) -> str:
    """Convert SHAP array to a plain-English explanation sentence."""
    risk_pct = prob * 100
    level    = "LOW" if risk_pct < 40 else "MEDIUM" if risk_pct < 70 else "HIGH"

    paired = sorted(zip(shap_vals, FEATURE_NAMES), key=lambda x: abs(x[0]), reverse=True)
    top_pos = [(v, n) for v, n in paired if v > 0][:3]
    top_neg = [(v, n) for v, n in paired if v < 0][:2]

    lbl = lambda n: FEATURE_LABELS.get(n, n.replace("_", " ").title())
    parts = [f"The model predicts {level} risk with a probability of {risk_pct:.1f}%."]

    if top_pos:
        factors = ", ".join(lbl(n) for _, n in top_pos)
        parts.append(f"Main risk factors: {factors}.")
    if top_neg:
        factors = ", ".join(lbl(n) for _, n in top_neg)
        parts.append(f"Positive factors: {factors}.")

    # Specific human-readable additions
    sv = dict(zip(FEATURE_NAMES, shap_vals))
    if sv.get("missed_payments", 0) > 0.05:
        parts.append("Missed payments are significantly increasing your risk.")
    if sv.get("credit_utilization", 0) > 0.05:
        parts.append("High credit utilization is a major risk factor.")
    if sv.get("total_dti", 0) > 0.05:
        parts.append("High total debt burden relative to income is concerning.")

    return " ".join(parts)


def shap_bar_b64(shap_vals: np.ndarray) -> str:
    """Generate a horizontal SHAP bar chart and return as base64 PNG."""
    labels = [FEATURE_LABELS.get(n, n) for n in FEATURE_NAMES]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_vals]
    idx    = np.argsort(np.abs(shap_vals))[-10:]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.barh([labels[i] for i in idx], [shap_vals[i] for i in idx],
            color=[colors[i] for i in idx], edgecolor="none", height=0.65)
    ax.axvline(0, color="#475569", linewidth=0.8, linestyle="--")
    ax.set_xlabel("SHAP Value", color="#94a3b8", fontsize=9)
    ax.set_title("Feature Impact on Risk Prediction", color="#e2e8f0", fontsize=11, fontweight="bold")
    ax.tick_params(colors="#e2e8f0", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

    ax.legend(handles=[
        mpatches.Patch(color="#ef4444", label="↑ Increases risk"),
        mpatches.Patch(color="#22c55e", label="↓ Decreases risk"),
    ], fontsize=8, facecolor="#1e293b", labelcolor="#e2e8f0", edgecolor="#334155")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
