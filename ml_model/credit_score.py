"""
ml_model/credit_score.py
========================
Computes a simulated credit score (300–900) using weighted rules.
User never enters their score — the system calculates it.

Weight breakdown:
  Payment History    35%
  Credit Utilization 30%
  Credit History     15%
  Income Stability   10%
  DTI / Burden       10%
"""


def compute_credit_score(
    missed_payments: int,
    credit_utilization: float,
    credit_history_years: float,
    monthly_income: float,
    employment_type: str,
    dti: float,
    loan_amount: float,
) -> dict:

    # 1. Payment history (35%)
    scores_by_missed = {0: 100, 1: 68, 2: 42, 3: 22, 4: 10}
    pay_score = scores_by_missed.get(min(missed_payments, 4), 5)

    # 2. Credit utilization (30%)
    u = credit_utilization
    if   u < 10:  util_score = 100
    elif u < 20:  util_score = 90
    elif u < 30:  util_score = 78
    elif u < 45:  util_score = 60
    elif u < 60:  util_score = 40
    elif u < 75:  util_score = 22
    else:         util_score = 8

    # 3. Credit history length (15%)
    h = credit_history_years
    if   h >= 15: hist_score = 100
    elif h >= 10: hist_score = 88
    elif h >=  7: hist_score = 74
    elif h >=  4: hist_score = 58
    elif h >=  2: hist_score = 40
    elif h >=  1: hist_score = 24
    else:         hist_score = 10

    # 4. Income stability (10%)
    emp_map = {"salaried": 100, "self-employed": 68, "unemployed": 15}
    inc_score = emp_map.get(employment_type.lower(), 50)
    if monthly_income >= 100_000: inc_score = min(100, inc_score + 8)
    elif monthly_income < 20_000: inc_score = max(0, inc_score - 20)

    # 5. DTI burden (10%)
    if   dti < 0.15: dti_score = 100
    elif dti < 0.25: dti_score = 84
    elif dti < 0.35: dti_score = 66
    elif dti < 0.50: dti_score = 46
    elif dti < 0.65: dti_score = 26
    else:            dti_score = 10

    composite = (
        pay_score  * 0.35 +
        util_score * 0.30 +
        hist_score * 0.15 +
        inc_score  * 0.10 +
        dti_score  * 0.10
    )

    score = max(300, min(900, int(300 + composite * 6.0)))

    if   score < 580: label = "Very Poor"
    elif score < 660: label = "Poor"
    elif score < 720: label = "Fair"
    elif score < 780: label = "Good"
    elif score < 850: label = "Very Good"
    else:             label = "Excellent"

    return {
        "score": score,
        "label": label,
        "breakdown": {
            "payment_history":    round(pay_score),
            "credit_utilization": round(util_score),
            "credit_history":     round(hist_score),
            "income_stability":   round(inc_score),
            "dti_burden":         round(dti_score),
        },
    }
