"""
backend/app.py
==============
Flask application — all API routes.

Routes
------
  POST /api/register
  POST /api/login
  POST /api/predict           [login_required]
  GET  /api/my-applications   [login_required]
  GET  /api/application/<id>  [login_required]

  GET  /api/admin/dashboard          [admin_required]
  GET  /api/admin/applications        [admin_required]
  POST /api/admin/decision/<id>       [admin_required]
  GET  /api/admin/decision-history/<id> [admin_required]
  GET  /api/admin/users               [admin_required]
  GET  /api/metrics                   [admin_required]
  GET  /api/model-comparison          [admin_required]
  POST /api/admin/train               [admin_required]
  GET  /api/health
"""

import json
import os
import sys
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Adjust path so ml_model and other modules are importable
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from backend.database import (
    init_db, save_application, get_application,
    get_user_applications, get_all_applications,
    admin_decide, get_decision_history,
    get_dashboard_stats, get_all_users,
)
from backend.auth import (
    register_service, login_service,
    login_required, admin_required,
)
from ml_model.feature_engineering import compute_features, to_array, scale
from ml_model.credit_score import compute_credit_score
from ml_model.decision_engine import make_decision
from ml_model.explain import approx_shap, shap_to_text, shap_bar_b64
from ml_model.recommendations import generate_recommendations
from ml_model.predictor import predict_all

app = Flask(__name__)
CORS(app)

# Initialise database on startup
init_db()


# ─── Helpers ────────────────────────────────────────────────────────────────

def ok(data=None, **kwargs):
    resp = {"success": True}
    if data is not None:
        resp.update(data if isinstance(data, dict) else {"data": data})
    resp.update(kwargs)
    return jsonify(resp), 200


def err(msg, code=400):
    return jsonify({"success": False, "error": msg}), code


# ─── Public routes ───────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return ok({"status": "running", "version": "2.0"})


@app.post("/api/register")
def register():
    body = request.get_json() or {}
    name     = body.get("name", "").strip()
    email    = body.get("email", "").strip().lower()
    password = body.get("password", "")
    if not name:
        return err("Name is required.")
    result = register_service(name, email, password)
    return ok({"message": result["msg"]}) if result["ok"] else err(result["msg"])


@app.post("/api/login")
def login():
    body = request.get_json() or {}
    result = login_service(
        body.get("email", "").strip().lower(),
        body.get("password", ""),
    )
    return ok(result) if result["ok"] else err(result["msg"], 401)


# ─── User routes ─────────────────────────────────────────────────────────────

@app.post("/api/predict")
@login_required
def predict():
    """
    Full pipeline:
      raw inputs → features → credit score → ML models → decision → SHAP → recommendations
    Status is always set to 'Pending' — admin decides later.
    """
    body = request.get_json() or {}

    # Validate required fields
    required = ["age", "employment_type", "monthly_income", "existing_emis",
                "loan_amount", "loan_tenure", "missed_payments",
                "credit_utilization", "credit_history_years"]
    for field in required:
        if field not in body:
            return err(f"Missing field: {field}")

    try:
        # 1. Feature engineering
        feats = compute_features(body)

        # 2. Credit score (internally computed, not from user)
        cs = compute_credit_score(
            missed_payments=int(body["missed_payments"]),
            credit_utilization=float(body["credit_utilization"]),
            credit_history_years=float(body["credit_history_years"]),
            monthly_income=float(body["monthly_income"]),
            employment_type=str(body["employment_type"]),
            dti=feats["dti"],
            loan_amount=float(body["loan_amount"]),
        )

        # 3. ML prediction (CNN + LSTM + TabTransformer ensemble)
        raw_arr  = to_array(feats)
        scaled   = scale(raw_arr)
        preds    = predict_all(scaled)
        prob     = preds["ensemble"]

        # 4. ML-based recommendation (not final decision — admin decides)
        ml_rec = "Recommend: Approve" if prob < 0.50 else "Recommend: Reject"

        # 5. Decision engine (used internally for reason codes)
        dec_result = make_decision(prob, cs["score"], feats)

        # 6. SHAP explanation
        shap_vals = approx_shap(scaled, prob)
        shap_text = shap_to_text(shap_vals, prob)
        shap_img  = shap_bar_b64(shap_vals)

        # 7. Recommendations
        recs = generate_recommendations(feats, prob, cs["score"], dec_result["decision"])

        # 8. Save to DB with status = Pending
        app_id = save_application({
            "user_id":          int(g.user["sub"]),
            "user_name":        g.user["name"],
            "user_email":       g.user["email"],
            "input_json":       json.dumps(body),
            "credit_score":     cs["score"],
            "score_label":      cs["label"],
            "probability":      round(prob, 4),
            "risk_category":    dec_result["risk_category"],
            "ml_recommendation":ml_rec,
            "reason_codes":     "; ".join(dec_result["reason_codes"]),
            "shap_explanation": shap_text,
            "recommendations":  json.dumps(recs),
        })

        return ok({
            "application_id":    app_id,
            "status":            "Pending",
            "credit_score":      cs["score"],
            "score_label":       cs["label"],
            "score_breakdown":   cs["breakdown"],
            "probability":       round(prob, 4),
            "probability_pct":   f"{prob*100:.1f}%",
            "risk_category":     dec_result["risk_category"],
            "ml_recommendation": ml_rec,
            "model_predictions": preds,
            "reason_codes":      dec_result["reason_codes"],
            "shap_explanation":  shap_text,
            "shap_chart_b64":    shap_img,
            "recommendations":   recs,
        })

    except Exception as e:
        app.logger.error(f"Predict error: {e}", exc_info=True)
        return err(f"Prediction failed: {str(e)}", 500)


@app.get("/api/my-applications")
@login_required
def my_applications():
    apps = get_user_applications(int(g.user["sub"]))
    for a in apps:
        _parse_json_fields(a)
    return ok({"applications": apps})


@app.get("/api/application/<int:app_id>")
@login_required
def get_one(app_id):
    appl = get_application(app_id)
    if not appl:
        return err("Application not found.", 404)
    # Users can only see their own applications
    if g.user["role"] != "admin" and int(g.user["sub"]) != appl["user_id"]:
        return err("Access denied.", 403)
    _parse_json_fields(appl)
    appl["decision_history"] = get_decision_history(app_id)
    return ok(appl)


# ─── Admin routes ─────────────────────────────────────────────────────────────

@app.get("/api/admin/dashboard")
@admin_required
def admin_dashboard():
    stats = get_dashboard_stats()
    # Attach model metrics if available
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            stats["model_metrics"] = json.load(f)
    return ok(stats)


@app.get("/api/admin/applications")
@admin_required
def admin_applications():
    status_filter = request.args.get("status")
    risk_filter   = request.args.get("risk")
    apps = get_all_applications(status_filter, risk_filter)
    for a in apps:
        _parse_json_fields(a)
    return ok({"applications": apps, "total": len(apps)})


@app.post("/api/admin/decision/<int:app_id>")
@admin_required
def admin_decision(app_id):
    """
    Admin approves or rejects an application.
    Body: { "status": "Approved" | "Rejected" | "Pending", "note": "..." }
    """
    body       = request.get_json() or {}
    new_status = body.get("status", "")
    note       = body.get("note", "")

    if new_status not in ("Approved", "Rejected", "Pending"):
        return err("Status must be 'Approved', 'Rejected', or 'Pending'.")

    success = admin_decide(app_id, new_status, g.user["email"], note)
    if not success:
        return err("Application not found.", 404)

    msg_map = {
        "Approved": f"Application #{app_id} has been approved.",
        "Rejected": f"Application #{app_id} has been rejected.",
        "Pending":  f"Application #{app_id} has been reset to Pending.",
    }
    app.logger.info(f"Admin {g.user['email']} → App #{app_id} → {new_status}")
    return ok({"message": msg_map[new_status], "new_status": new_status})


@app.get("/api/admin/decision-history/<int:app_id>")
@admin_required
def decision_history(app_id):
    history = get_decision_history(app_id)
    return ok({"history": history})


@app.get("/api/admin/users")
@admin_required
def admin_users():
    users = get_all_users()
    return ok({"users": users})


@app.get("/api/metrics")
@admin_required
def get_metrics():
    path = os.path.join(BASE_DIR, "models", "metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return ok(json.load(f))
    return ok({"message": "Models not trained yet. Run train_model.py first."})


@app.get("/api/model-comparison")
@admin_required
def model_comparison():
    path = os.path.join(BASE_DIR, "models", "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return ok(json.load(f))
    return ok({"message": "Run train_model.py to generate comparison data."})


@app.post("/api/admin/train")
@admin_required
def trigger_train():
    """Admin can trigger model retraining."""
    try:
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "train_model.py")],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode == 0:
            return ok({"message": "Training complete!", "output": result.stdout[-1500:]})
        return err(f"Training failed:\n{result.stderr[-1500:]}", 500)
    except Exception as e:
        return err(f"Error: {str(e)}", 500)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_json_fields(app_dict):
    """Parse JSON string fields to Python objects in-place."""
    for field in ["recommendations", "input_json"]:
        if app_dict.get(field) and isinstance(app_dict[field], str):
            try:
                app_dict[field] = json.loads(app_dict[field])
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
