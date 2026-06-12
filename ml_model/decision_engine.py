"""
ml_model/decision_engine.py
============================
Combines ML probability + credit score to produce:
  - risk_category (LOW / MEDIUM / HIGH)
  - internal decision suggestion (used for reason codes only)
  - reason_codes (list of human-readable risk factors)

NOTE: The final approve/reject decision is ALWAYS made by the admin.
      This engine only provides an ML-based recommendation.
"""

from typing import List


REASON_MAP = {
    "high_dti":      "Your total debt-to-income ratio is too high (above 50%).",
    "low_income":    "Monthly income is below the recommended threshold.",
    "high_util":     "Credit card utilization is very high (above 60%).",
    "short_history": "Your credit history is too short (under 2 years).",
    "missed_pay":    "You have multiple missed payments in the last 12 months.",
    "low_score":     "Credit score is below the minimum acceptable level.",
    "unemployed":    "No confirmed employment — stable income is required.",
    "large_loan":    "Loan amount is very high relative to your income.",
}


def make_decision(probability: float, credit_score: int, features: dict) -> dict:
    """
    Parameters
    ----------
    probability  : ensemble ML probability (0–1)
    credit_score : simulated credit score (300–900)
    features     : engineered feature dict

    Returns
    -------
    dict with risk_category, decision (internal), reason_codes
    """
    prob = float(probability)

    # Risk category
    if   prob < 0.40: risk_category = "LOW"
    elif prob < 0.70: risk_category = "MEDIUM"
    else:             risk_category = "HIGH"

    # Collect reason codes
    reasons: List[str] = []
    if features.get("total_dti", 0) > 0.50:
        reasons.append(REASON_MAP["high_dti"])
    if features.get("monthly_income", 0) < 20_000:
        reasons.append(REASON_MAP["low_income"])
    if features.get("credit_utilization", 0) > 60:
        reasons.append(REASON_MAP["high_util"])
    if features.get("credit_history_years", 0) < 2:
        reasons.append(REASON_MAP["short_history"])
    if features.get("missed_payments", 0) >= 2:
        reasons.append(REASON_MAP["missed_pay"])
    if credit_score < 600:
        reasons.append(REASON_MAP["low_score"])
    if features.get("employment_encoded", 0) == 2:
        reasons.append(REASON_MAP["unemployed"])
    if features.get("income_loan_ratio", 1) < 0.03:
        reasons.append(REASON_MAP["large_loan"])

    # Internal decision (only for reason codes / logic — not shown to user as final)
    if prob < 0.30 and credit_score >= 700:
        decision = "APPROVE"
    elif prob > 0.60 or credit_score < 580:
        decision = "REJECT"
    else:
        decision = "REVIEW"

    return {
        "risk_category": risk_category,
        "decision":      decision,
        "reason_codes":  reasons if reasons else ["No significant risk factors detected."],
    }
