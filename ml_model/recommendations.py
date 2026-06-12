"""
ml_model/recommendations.py
============================
Generates personalised, actionable improvement advice based on:
  - engineered features
  - ML probability
  - credit score
  - internal decision
"""

from typing import List, Dict


def generate_recommendations(
    features: dict,
    probability: float,
    credit_score: int,
    decision: str,
) -> List[Dict]:
    """
    Returns a list of recommendation dicts, sorted by priority.

    Each dict has: priority, title, detail, impact
    """
    recs = []

    util   = features.get("credit_utilization", 0)
    missed = features.get("missed_payments", 0)
    tdti   = features.get("total_dti", 0)
    dti    = features.get("dti", 0)
    hist   = features.get("credit_history_years", 0)
    income = features.get("monthly_income", 0)
    loan   = features.get("loan_amount", 0)
    tenure = features.get("loan_tenure", 0)
    emp    = features.get("employment_encoded", 0)
    incr   = features.get("income_loan_ratio", 0)

    # 1. Missed payments
    if missed >= 3:
        recs.append({"priority": "high", "title": "Fix Missed Payments Immediately",
            "detail": f"You have {missed} missed payments — the single biggest red flag for lenders. "
                      "Set up auto-pay to ensure on-time payments going forward.",
            "impact": "Can improve credit score by 50–100 points over 6–12 months."})
    elif missed >= 1:
        recs.append({"priority": "medium", "title": "Avoid Further Missed Payments",
            "detail": "Even one missed payment hurts significantly. "
                      "Maintain a perfect payment record for the next 6 months before reapplying.",
            "impact": "Clean 6-month payment history can raise score 20–40 points."})

    # 2. Credit utilization
    if util > 75:
        recs.append({"priority": "high", "title": "Reduce Credit Card Utilization",
            "detail": f"Your utilization is {util:.0f}% — critically high. "
                      "Pay down card balances to below 30% before applying.",
            "impact": f"Reducing from {util:.0f}% to 30% can boost score 60–120 points."})
    elif util > 40:
        recs.append({"priority": "medium", "title": "Lower Credit Utilization",
            "detail": f"At {util:.0f}%, you're using too much of your available credit. "
                      "Target under 30% for a much better score.",
            "impact": "30–60 point improvement possible."})

    # 3. Debt burden
    if tdti > 0.65:
        recs.append({"priority": "high", "title": "Reduce Total Debt Burden",
            "detail": f"After this loan, your monthly obligations would consume {tdti*100:.0f}% "
                      "of income. Pay off existing EMIs first, or request a smaller loan amount.",
            "impact": "Getting DTI below 40% is essential for approval."})
    elif tdti > 0.50:
        recs.append({"priority": "high", "title": "High Debt-to-Income Ratio",
            "detail": f"Total obligations would reach {tdti*100:.0f}% of income. "
                      f"Consider extending tenure to {tenure+24} months to reduce the EMI.",
            "impact": "Reducing monthly burden below 40% strongly improves chances."})

    # 4. Loan size
    if incr < 0.03:
        recs.append({"priority": "high", "title": "Reduce Requested Loan Amount",
            "detail": f"Requesting ₹{loan:,.0f} on ₹{income:,.0f}/month income is very aggressive. "
                      "Try requesting 40–60% of this amount instead.",
            "impact": "Smaller loans have much higher approval rates for your income level."})

    # 5. Credit history
    if hist < 1:
        recs.append({"priority": "medium", "title": "Build Your Credit History",
            "detail": "Less than 1 year of credit history makes lenders nervous. "
                      "Keep existing accounts active; a secured card used monthly builds history quickly.",
            "impact": "After 2+ years of history, score improves 30–50 points."})
    elif hist < 2.5:
        recs.append({"priority": "low", "title": "Allow Credit History to Grow",
            "detail": "Keep old credit accounts open even if unused. "
                      "Do not close cards — history length matters.",
            "impact": "Time and consistent behaviour is the key factor here."})

    # 6. Employment / income
    if emp == 2:
        recs.append({"priority": "high", "title": "Secure Employment or Add Co-Applicant",
            "detail": "Lenders require proof of stable income. Without employment, "
                      "approval is very difficult. A co-applicant with steady income can help.",
            "impact": "A co-applicant can transform a rejection into an approval."})
    elif income < 20_000:
        recs.append({"priority": "medium", "title": "Increase Income or Lower Loan",
            "detail": f"Income of ₹{income:,.0f}/month is below recommended levels for this loan. "
                      "Request a lower amount or consider a co-applicant.",
            "impact": "20% income improvement significantly expands your eligible loan range."})

    # 7. Opening message
    if decision == "APPROVE":
        recs.insert(0, {"priority": "info", "title": "✅ Looking Good!",
            "detail": "Your profile looks eligible for approval. "
                      "Maintain these habits — consistent payments and low utilization keep your score healthy.",
            "impact": "Continued good behaviour qualifies you for better rates in future."})
    elif decision == "REVIEW":
        recs.insert(0, {"priority": "info", "title": "🔍 Close to Approval",
            "detail": "Your application is borderline. Addressing 1–2 HIGH priority items "
                      "and reapplying in 3–6 months could secure direct approval.",
            "impact": ""})
    else:
        recs.insert(0, {"priority": "info", "title": "📋 Here's Your Improvement Plan",
            "detail": "Your application cannot be approved right now. "
                      "Following the steps below for 6–12 months should move you into the eligible range.",
            "impact": ""})

    # Sort: high → medium → low → info
    order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    recs.sort(key=lambda r: order.get(r["priority"], 4))
    return recs
