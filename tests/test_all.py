"""
tests/test_all.py
=================
Full test suite for CreditIQ v2.

Covers:
  - Credit score engine
  - Feature engineering
  - Decision engine
  - Recommendation engine
  - Auth (password hashing, JWT)
  - SHAP approximation
  - Database operations (in-memory)
  - Flask API endpoints (integration tests)

Run:  pytest tests/ -v --tb=short
Run with coverage:  pytest tests/ -v --cov=. --cov-report=term-missing
"""

import sys
import os
import json
import pytest
import numpy as np

# Add project root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Credit Score Engine
# ─────────────────────────────────────────────────────────────────────────────

from ml_model.credit_score import compute_credit_score
from frontend.config import get_api_base


def test_get_api_base_prefers_api_url_env(monkeypatch):
    monkeypatch.setenv("API_URL", "https://backend.example.com")
    monkeypatch.delenv("BACKEND_URL", raising=False)
    assert get_api_base() == "https://backend.example.com"


def test_get_api_base_falls_back_to_localhost(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.delenv("RENDER", raising=False)
    assert get_api_base() == "http://127.0.0.1:5000"


def test_get_api_base_uses_render_backend_fallback(monkeypatch):
    monkeypatch.delenv("API_URL", raising=False)
    monkeypatch.delenv("BACKEND_URL", raising=False)
    monkeypatch.setenv("RENDER", "true")
    assert get_api_base() == "https://credit-risk-api.onrender.com"


class TestCreditScore:
    """Tests for the rule-based credit score engine."""

    def _score(self, **overrides):
        """Helper: build a default 'good borrower' profile and apply overrides."""
        defaults = dict(
            missed_payments=0,
            credit_utilization=20.0,
            credit_history_years=5.0,
            monthly_income=60000.0,
            employment_type="salaried",
            dti=0.20,
            loan_amount=500000.0,
        )
        defaults.update(overrides)
        return compute_credit_score(**defaults)

    def test_perfect_profile_is_good_or_excellent(self):
        r = self._score(missed_payments=0, credit_utilization=10,
                        credit_history_years=12, monthly_income=120000, dti=0.10)
        assert r["score"] >= 750
        assert r["label"] in ("Very Good", "Excellent")

    def test_score_always_in_300_to_900(self):
        """Score must never go outside 300–900 regardless of inputs."""
        for _ in range(30):
            r = self._score(
                missed_payments=np.random.randint(0, 5),
                credit_utilization=np.random.uniform(0, 100),
                credit_history_years=np.random.uniform(0, 25),
                monthly_income=np.random.uniform(5000, 300000),
                dti=np.random.uniform(0, 1.5),
            )
            assert 300 <= r["score"] <= 900, f"Score out of range: {r['score']}"

    def test_more_missed_payments_lower_score(self):
        score_0 = self._score(missed_payments=0)["score"]
        score_3 = self._score(missed_payments=3)["score"]
        assert score_3 < score_0, "3 misses should score lower than 0 misses"

    def test_high_utilization_lower_score(self):
        low  = self._score(credit_utilization=8)["score"]
        high = self._score(credit_utilization=90)["score"]
        assert high < low

    def test_unemployed_lower_than_salaried(self):
        sal = self._score(employment_type="salaried")["score"]
        une = self._score(employment_type="unemployed")["score"]
        assert une < sal

    def test_short_history_lower_score(self):
        long_  = self._score(credit_history_years=15)["score"]
        short_ = self._score(credit_history_years=0.5)["score"]
        assert short_ < long_

    def test_label_matches_score_range(self):
        r = self._score()
        score = r["score"]
        label = r["label"]
        if score < 580:   assert label == "Very Poor"
        elif score < 660: assert label == "Poor"
        elif score < 720: assert label == "Fair"
        elif score < 780: assert label == "Good"
        elif score < 850: assert label == "Very Good"
        else:             assert label == "Excellent"

    def test_breakdown_has_all_five_components(self):
        r = self._score()
        expected = {
            "payment_history", "credit_utilization",
            "credit_history", "income_stability", "dti_burden"
        }
        assert set(r["breakdown"].keys()) == expected

    def test_breakdown_values_0_to_100(self):
        r = self._score()
        for k, v in r["breakdown"].items():
            assert 0 <= v <= 100, f"Breakdown component {k}={v} out of range"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

from ml_model.feature_engineering import compute_features, to_array, scale


class TestFeatureEngineering:
    BASE = dict(
        age=32, employment_type="salaried", monthly_income=60000.0,
        existing_emis=10000.0, loan_amount=500000.0, loan_tenure=60,
        missed_payments=1, credit_utilization=35.0, credit_history_years=4.0,
    )

    def test_dti_is_emis_over_income(self):
        f = compute_features(self.BASE)
        expected = 10000.0 / 60000.0
        assert abs(f["dti"] - expected) < 0.001

    def test_dti_capped_at_2(self):
        raw = dict(self.BASE, existing_emis=500000, monthly_income=1000)
        f   = compute_features(raw)
        assert f["dti"] <= 2.0, "DTI should be capped at 2.0"

    def test_feature_array_has_13_elements(self):
        f   = compute_features(self.BASE)
        arr = to_array(f)
        assert len(arr) == 13

    def test_employment_encoding(self):
        assert compute_features(dict(self.BASE, employment_type="salaried"))["employment_encoded"]     == 0.0
        assert compute_features(dict(self.BASE, employment_type="self-employed"))["employment_encoded"] == 1.0
        assert compute_features(dict(self.BASE, employment_type="unemployed"))["employment_encoded"]    == 2.0

    def test_total_dti_includes_new_emi(self):
        f = compute_features(self.BASE)
        # total_dti must be >= dti (adds new loan EMI on top)
        assert f["total_dti"] >= f["dti"]

    def test_income_loan_ratio_positive(self):
        f = compute_features(self.BASE)
        assert f["income_loan_ratio"] > 0

    def test_array_is_float32(self):
        f   = compute_features(self.BASE)
        arr = to_array(f)
        assert arr.dtype == np.float32

    def test_scale_returns_array_of_same_length(self):
        f   = compute_features(self.BASE)
        arr = to_array(f)
        scaled = scale(arr)
        assert len(scaled) == 13


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Decision Engine
# ─────────────────────────────────────────────────────────────────────────────

from ml_model.decision_engine import make_decision


class TestDecisionEngine:
    SAFE_FEATS = dict(
        age=35, monthly_income=80000, loan_amount=500000, loan_tenure=60,
        existing_emis=5000, missed_payments=0, credit_utilization=20,
        credit_history_years=8, dti=0.0625, total_dti=0.18,
        income_loan_ratio=0.16, employment_encoded=0, emi_new=11000,
    )

    def test_approve_condition(self):
        r = make_decision(0.15, 780, self.SAFE_FEATS)
        assert r["decision"] == "APPROVE"

    def test_reject_high_probability(self):
        r = make_decision(0.85, 750, self.SAFE_FEATS)
        assert r["decision"] == "REJECT"

    def test_reject_low_credit_score(self):
        r = make_decision(0.20, 520, self.SAFE_FEATS)
        assert r["decision"] == "REJECT"

    def test_review_borderline(self):
        r = make_decision(0.50, 660, self.SAFE_FEATS)
        assert r["decision"] == "REVIEW"

    def test_risk_low(self):
        assert make_decision(0.20, 780, self.SAFE_FEATS)["risk_category"] == "LOW"

    def test_risk_medium(self):
        assert make_decision(0.55, 700, self.SAFE_FEATS)["risk_category"] == "MEDIUM"

    def test_risk_high(self):
        assert make_decision(0.80, 700, self.SAFE_FEATS)["risk_category"] == "HIGH"

    def test_result_has_required_keys(self):
        r = make_decision(0.50, 660, self.SAFE_FEATS)
        for k in ("decision", "risk_category", "reason_codes"):
            assert k in r, f"Key '{k}' missing from decision result"

    def test_reason_codes_is_list(self):
        r = make_decision(0.50, 660, self.SAFE_FEATS)
        assert isinstance(r["reason_codes"], list)

    def test_high_dti_adds_reason_code(self):
        risky = dict(self.SAFE_FEATS, total_dti=0.75)
        r = make_decision(0.70, 600, risky)
        assert any("debt-to-income" in rc.lower() for rc in r["reason_codes"])

    def test_high_utilization_adds_reason_code(self):
        risky = dict(self.SAFE_FEATS, credit_utilization=80)
        r = make_decision(0.70, 600, risky)
        assert any("utilization" in rc.lower() for rc in r["reason_codes"])

    def test_missed_payments_adds_reason_code(self):
        risky = dict(self.SAFE_FEATS, missed_payments=3)
        r = make_decision(0.70, 600, risky)
        assert any("missed" in rc.lower() for rc in r["reason_codes"])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Recommendation Engine
# ─────────────────────────────────────────────────────────────────────────────

from ml_model.recommendations import generate_recommendations


class TestRecommendations:
    BASE = dict(
        age=35, monthly_income=55000, loan_amount=500000, loan_tenure=60,
        existing_emis=8000, missed_payments=0, credit_utilization=30,
        credit_history_years=5, dti=0.145, total_dti=0.30,
        income_loan_ratio=0.11, employment_encoded=0, emi_new=11000,
    )

    def test_returns_non_empty_list(self):
        recs = generate_recommendations(self.BASE, 0.30, 720, "APPROVE")
        assert isinstance(recs, list)
        assert len(recs) > 0

    def test_every_rec_has_required_keys(self):
        recs = generate_recommendations(self.BASE, 0.30, 720, "APPROVE")
        for rec in recs:
            for key in ("priority", "title", "detail"):
                assert key in rec, f"Key '{key}' missing from recommendation"

    def test_valid_priority_values(self):
        recs = generate_recommendations(self.BASE, 0.80, 520, "REJECT")
        for rec in recs:
            assert rec["priority"] in ("high", "medium", "low", "info")

    def test_high_missed_payments_triggers_high_priority(self):
        feats = dict(self.BASE, missed_payments=4)
        recs  = generate_recommendations(feats, 0.80, 540, "REJECT")
        high_titles = [r["title"] for r in recs if r["priority"] == "high"]
        assert any("payment" in t.lower() or "miss" in t.lower() for t in high_titles)

    def test_unemployed_triggers_employment_rec(self):
        feats = dict(self.BASE, employment_encoded=2)
        recs  = generate_recommendations(feats, 0.85, 490, "REJECT")
        titles = [r["title"].lower() for r in recs]
        assert any("employ" in t or "income" in t for t in titles)

    def test_high_utilization_triggers_rec(self):
        feats = dict(self.BASE, credit_utilization=88)
        recs  = generate_recommendations(feats, 0.75, 560, "REJECT")
        titles = [r["title"].lower() for r in recs]
        assert any("utilization" in t or "credit card" in t for t in titles)

    def test_high_priority_before_low_priority(self):
        feats = dict(self.BASE, missed_payments=4, credit_utilization=85, total_dti=0.8)
        recs  = generate_recommendations(feats, 0.92, 480, "REJECT")
        priorities = [r["priority"] for r in recs]
        order_map  = {"high": 0, "medium": 1, "low": 2, "info": 3}
        ordered    = [order_map.get(p, 4) for p in priorities]
        assert ordered == sorted(ordered), "Recommendations not sorted by priority"

    def test_approve_decision_has_positive_opening(self):
        recs = generate_recommendations(self.BASE, 0.20, 760, "APPROVE")
        first = recs[0]
        assert "good" in first["title"].lower() or "✅" in first["title"] or "approv" in first["title"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Authentication
# ─────────────────────────────────────────────────────────────────────────────

from backend.auth import hash_pw, verify_pw, make_token, decode_token


class TestAuth:

    def test_password_hash_is_different_from_plaintext(self):
        pw = "MyPassword123"
        h  = hash_pw(pw)
        assert h != pw

    def test_correct_password_verifies(self):
        pw = "SecurePass@99"
        h  = hash_pw(pw)
        assert verify_pw(pw, h) is True

    def test_wrong_password_fails(self):
        h = hash_pw("correct")
        assert verify_pw("wrong", h) is False

    def test_two_hashes_of_same_password_are_different(self):
        """bcrypt uses random salt — same password = different hash."""
        pw = "samepassword"
        h1 = hash_pw(pw)
        h2 = hash_pw(pw)
        assert h1 != h2
        assert verify_pw(pw, h1)
        assert verify_pw(pw, h2)

    def test_jwt_roundtrip(self):
        token   = make_token(42, "test@example.com", "user", "Test User")
        payload = decode_token(token)
        assert payload is not None
        assert payload["email"] == "test@example.com"
        assert payload["role"]  == "user"
        assert payload["sub"]   == "42"

    def test_invalid_jwt_returns_none(self):
        assert decode_token("this.is.not.valid") is None

    def test_tampered_jwt_returns_none(self):
        token    = make_token(1, "a@b.com", "user", "A")
        tampered = token[:-10] + "XXXXXXXXXX"
        assert decode_token(tampered) is None

    def test_admin_token_contains_admin_role(self):
        token   = make_token(1, "admin@bank.com", "admin", "Admin")
        payload = decode_token(token)
        assert payload["role"] == "admin"


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SHAP Explainability
# ─────────────────────────────────────────────────────────────────────────────

from ml_model.explain import approx_shap, shap_to_text


class TestExplainability:

    def test_shap_output_shape_matches_features(self):
        feats  = np.random.randn(13).astype(np.float32)
        shap_v = approx_shap(feats, 0.75)
        assert shap_v.shape == (13,)

    def test_high_risk_produces_positive_shap_sum(self):
        feats = np.ones(13, dtype=np.float32)
        sv    = approx_shap(feats, 0.90)
        assert sv.sum() > 0, "High risk prob should produce net positive SHAP"

    def test_low_risk_produces_negative_shap_sum(self):
        feats = np.ones(13, dtype=np.float32)
        sv    = approx_shap(feats, 0.10)
        assert sv.sum() < 0, "Low risk prob should produce net negative SHAP"

    def test_shap_text_is_non_empty_string(self):
        feats = np.random.randn(13).astype(np.float32)
        sv    = approx_shap(feats, 0.80)
        text  = shap_to_text(sv, 0.80)
        assert isinstance(text, str)
        assert len(text) > 20

    def test_shap_text_mentions_risk_level(self):
        feats = np.random.randn(13).astype(np.float32)
        sv    = approx_shap(feats, 0.80)
        text  = shap_to_text(sv, 0.80)
        assert any(level in text for level in ("LOW", "MEDIUM", "HIGH"))

    def test_shap_text_includes_probability(self):
        feats = np.random.randn(13).astype(np.float32)
        sv    = approx_shap(feats, 0.65)
        text  = shap_to_text(sv, 0.65)
        assert "65.0%" in text


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Database Layer
# ─────────────────────────────────────────────────────────────────────────────

import tempfile
import sqlite3

# Patch DB_PATH to use a temp file for tests
import backend.database as db_module


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """Each test gets an isolated temp database."""
    test_db = str(tmp_path / "test.db")
    monkeypatch.setattr(db_module, "DB_PATH", test_db)
    db_module.init_db()
    return test_db


class TestDatabase:

    def test_init_creates_tables(self, temp_db):
        conn = sqlite3.connect(temp_db)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        conn.close()
        assert "users"        in tables
        assert "applications" in tables
        assert "decisions"    in tables

    def test_init_creates_default_admin(self, temp_db):
        conn = sqlite3.connect(temp_db)
        row  = conn.execute("SELECT email,role FROM users WHERE role='admin'").fetchone()
        conn.close()
        assert row is not None
        assert row[0] == "admin@bank.com"

    def test_create_user_and_retrieve(self, temp_db):
        import bcrypt
        pw_hash = bcrypt.hashpw(b"test123", bcrypt.gensalt()).decode()
        result  = db_module.create_user("Alice", "alice@test.com", pw_hash)
        assert result["ok"] is True

        user = db_module.get_user_by_email("alice@test.com")
        assert user is not None
        assert user["name"]  == "Alice"
        assert user["role"]  == "user"

    def test_duplicate_email_rejected(self, temp_db):
        import bcrypt
        pw = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        db_module.create_user("Bob", "bob@test.com", pw)
        result = db_module.create_user("Bob2", "bob@test.com", pw)
        assert result["ok"] is False
        assert "already registered" in result["msg"].lower()

    def test_save_and_retrieve_application(self, temp_db):
        import bcrypt
        pw = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        db_module.create_user("Carol", "carol@test.com", pw)
        user = db_module.get_user_by_email("carol@test.com")

        app_id = db_module.save_application({
            "user_id": user["id"], "user_name": "Carol",
            "user_email": "carol@test.com", "input_json": "{}",
            "credit_score": 720, "score_label": "Good",
            "probability": 0.30, "risk_category": "LOW",
            "ml_recommendation": "Recommend: Approve",
            "reason_codes": "", "shap_explanation": "Test explanation.",
            "recommendations": "[]",
        })
        assert app_id > 0

        apps = db_module.get_user_applications(user["id"])
        assert len(apps) == 1
        assert apps[0]["credit_score"] == 720
        assert apps[0]["status"]       == "Pending"

    def test_admin_decide_approve(self, temp_db):
        import bcrypt
        pw = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        db_module.create_user("Dave", "dave@test.com", pw)
        user   = db_module.get_user_by_email("dave@test.com")
        app_id = db_module.save_application({
            "user_id": user["id"], "user_name": "Dave",
            "user_email": "dave@test.com", "input_json": "{}",
            "credit_score": 750, "score_label": "Good",
            "probability": 0.25, "risk_category": "LOW",
            "ml_recommendation": "Recommend: Approve",
            "reason_codes": "", "shap_explanation": "",
            "recommendations": "[]",
        })

        result = db_module.admin_decide(app_id, "Approved", "admin@bank.com", "Looks good")
        assert result is True

        app = db_module.get_application(app_id)
        assert app["status"]      == "Approved"
        assert app["reviewed_by"] == "admin@bank.com"

    def test_admin_decide_reject(self, temp_db):
        import bcrypt
        pw = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        db_module.create_user("Eve", "eve@test.com", pw)
        user   = db_module.get_user_by_email("eve@test.com")
        app_id = db_module.save_application({
            "user_id": user["id"], "user_name": "Eve",
            "user_email": "eve@test.com", "input_json": "{}",
            "credit_score": 500, "score_label": "Very Poor",
            "probability": 0.85, "risk_category": "HIGH",
            "ml_recommendation": "Recommend: Reject",
            "reason_codes": "High risk", "shap_explanation": "",
            "recommendations": "[]",
        })

        db_module.admin_decide(app_id, "Rejected", "admin@bank.com", "Too risky")
        app = db_module.get_application(app_id)
        assert app["status"] == "Rejected"

    def test_decision_history_logged(self, temp_db):
        import bcrypt
        pw = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        db_module.create_user("Frank", "frank@test.com", pw)
        user   = db_module.get_user_by_email("frank@test.com")
        app_id = db_module.save_application({
            "user_id": user["id"], "user_name": "Frank",
            "user_email": "frank@test.com", "input_json": "{}",
            "credit_score": 650, "score_label": "Fair",
            "probability": 0.50, "risk_category": "MEDIUM",
            "ml_recommendation": "Recommend: Reject",
            "reason_codes": "", "shap_explanation": "",
            "recommendations": "[]",
        })

        db_module.admin_decide(app_id, "Approved", "admin@bank.com", "Manual override")
        db_module.admin_decide(app_id, "Rejected", "admin@bank.com", "Changed mind")

        history = db_module.get_decision_history(app_id)
        assert len(history) == 2
        assert history[0]["new_status"] == "Rejected"   # most recent first
        assert history[1]["new_status"] == "Approved"

    def test_get_all_applications_with_status_filter(self, temp_db):
        import bcrypt
        pw = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
        db_module.create_user("G", "g@test.com", pw)
        user = db_module.get_user_by_email("g@test.com")

        base = {"user_id": user["id"], "user_name": "G", "user_email": "g@test.com",
                "input_json": "{}", "credit_score": 700, "score_label": "Good",
                "probability": 0.30, "risk_category": "LOW",
                "ml_recommendation": "Recommend: Approve",
                "reason_codes": "", "shap_explanation": "", "recommendations": "[]"}

        id1 = db_module.save_application(base)
        id2 = db_module.save_application(base)
        db_module.admin_decide(id1, "Approved", "admin@bank.com", "")

        approved = db_module.get_all_applications(status_filter="Approved")
        pending  = db_module.get_all_applications(status_filter="Pending")

        assert any(a["id"] == id1 for a in approved)
        assert any(a["id"] == id2 for a in pending)
        assert not any(a["id"] == id1 for a in pending)

    def test_invalid_status_rejected_by_admin_decide(self, temp_db):
        result = db_module.admin_decide(999, "InvalidStatus", "admin@bank.com", "")
        assert result is False


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Flask API Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

import tempfile

@pytest.fixture
def flask_client(tmp_path, monkeypatch):
    """Flask test client with isolated temp database."""
    import backend.database as db_mod
    test_db = str(tmp_path / "api_test.db")
    monkeypatch.setattr(db_mod, "DB_PATH", test_db)

    import backend.app as flask_app_module
    flask_app_module.init_db()

    flask_app_module.app.config["TESTING"] = True
    with flask_app_module.app.test_client() as client:
        yield client


def _register_and_login(client, email, password, name="Test User", role="user"):
    """Helper: register + login and return token."""
    client.post("/api/register", json={"name": name, "email": email, "password": password})
    r = client.post("/api/login", json={"email": email, "password": password})
    data = r.get_json()
    return data.get("token")


LOAN_PAYLOAD = {
    "age": 30, "employment_type": "salaried",
    "monthly_income": 60000, "existing_emis": 8000,
    "loan_amount": 500000, "loan_tenure": 60,
    "missed_payments": 0, "credit_utilization": 25,
    "credit_history_years": 5,
}


class TestFlaskAPI:

    def test_health_check(self, flask_client):
        r = flask_client.get("/api/health")
        assert r.status_code == 200
        assert r.get_json()["status"] == "running"

    def test_register_success(self, flask_client):
        r = flask_client.post("/api/register",
                              json={"name": "Alice", "email": "alice@test.com",
                                    "password": "pass123"})
        assert r.status_code == 200
        assert r.get_json()["success"] is True

    def test_register_duplicate_email(self, flask_client):
        flask_client.post("/api/register",
                          json={"name": "A", "email": "dup@test.com", "password": "pass123"})
        r = flask_client.post("/api/register",
                              json={"name": "B", "email": "dup@test.com", "password": "pass456"})
        assert r.status_code == 400

    def test_register_short_password(self, flask_client):
        r = flask_client.post("/api/register",
                              json={"name": "C", "email": "c@test.com", "password": "12"})
        assert r.status_code == 400

    def test_login_success(self, flask_client):
        flask_client.post("/api/register",
                          json={"name": "D", "email": "d@test.com", "password": "pass123"})
        r = flask_client.post("/api/login",
                              json={"email": "d@test.com", "password": "pass123"})
        assert r.status_code == 200
        data = r.get_json()
        assert "token" in data
        assert data["user"]["role"] == "user"

    def test_login_wrong_password(self, flask_client):
        flask_client.post("/api/register",
                          json={"name": "E", "email": "e@test.com", "password": "correct"})
        r = flask_client.post("/api/login",
                              json={"email": "e@test.com", "password": "wrong"})
        assert r.status_code == 401

    def test_login_unknown_email(self, flask_client):
        r = flask_client.post("/api/login",
                              json={"email": "nobody@test.com", "password": "pass"})
        assert r.status_code == 401

    def test_predict_requires_auth(self, flask_client):
        r = flask_client.post("/api/predict", json=LOAN_PAYLOAD)
        assert r.status_code == 401

    def test_predict_success(self, flask_client):
        token = _register_and_login(flask_client, "f@test.com", "pass123", "F")
        r = flask_client.post("/api/predict", json=LOAN_PAYLOAD,
                              headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 200
        data = r.get_json()
        assert "application_id"    in data
        assert "credit_score"      in data
        assert "probability"       in data
        assert "risk_category"     in data
        assert "ml_recommendation" in data
        assert data["status"]      == "Pending"

    def test_predict_saves_to_db(self, flask_client):
        token = _register_and_login(flask_client, "g@test.com", "pass123", "G")
        flask_client.post("/api/predict", json=LOAN_PAYLOAD,
                          headers={"Authorization": f"Bearer {token}"})
        r = flask_client.get("/api/my-applications",
                             headers={"Authorization": f"Bearer {token}"})
        apps = r.get_json()["applications"]
        assert len(apps) == 1
        assert apps[0]["status"] == "Pending"

    def test_my_applications_requires_auth(self, flask_client):
        r = flask_client.get("/api/my-applications")
        assert r.status_code == 401

    def test_user_cannot_access_admin_dashboard(self, flask_client):
        token = _register_and_login(flask_client, "h@test.com", "pass123", "H")
        r = flask_client.get("/api/admin/dashboard",
                             headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_user_cannot_access_admin_applications(self, flask_client):
        token = _register_and_login(flask_client, "i@test.com", "pass123", "I")
        r = flask_client.get("/api/admin/applications",
                             headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_user_cannot_make_admin_decision(self, flask_client):
        token = _register_and_login(flask_client, "j@test.com", "pass123", "J")
        r = flask_client.post("/api/admin/decision/1",
                              json={"status": "Approved", "note": ""},
                              headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 403

    def test_admin_can_access_dashboard(self, flask_client):
        token = _register_and_login(flask_client, "admin@bank.com", "admin123",
                                    "Admin", "admin")
        r = flask_client.get("/api/admin/dashboard",
                             headers={"Authorization": f"Bearer {token}"})
        # Admin already seeded in init_db — just do login
        # Login as seeded admin
        r2 = flask_client.post("/api/login",
                               json={"email": "admin@bank.com", "password": "admin123"})
        admin_token = r2.get_json().get("token")
        r3 = flask_client.get("/api/admin/dashboard",
                              headers={"Authorization": f"Bearer {admin_token}"})
        assert r3.status_code == 200

    def test_admin_approve_application(self, flask_client):
        # User submits
        user_token = _register_and_login(flask_client, "k@test.com", "pass123", "K")
        pred_r = flask_client.post("/api/predict", json=LOAN_PAYLOAD,
                                   headers={"Authorization": f"Bearer {user_token}"})
        app_id = pred_r.get_json()["application_id"]

        # Admin approves
        r2 = flask_client.post("/api/login",
                               json={"email": "admin@bank.com", "password": "admin123"})
        admin_token = r2.get_json()["token"]
        r3 = flask_client.post(f"/api/admin/decision/{app_id}",
                               json={"status": "Approved", "note": "Approved after review"},
                               headers={"Authorization": f"Bearer {admin_token}"})
        assert r3.status_code == 200
        assert r3.get_json()["new_status"] == "Approved"

        # User sees updated status
        apps_r = flask_client.get("/api/my-applications",
                                  headers={"Authorization": f"Bearer {user_token}"})
        apps = apps_r.get_json()["applications"]
        assert apps[0]["status"] == "Approved"

    def test_admin_reject_application(self, flask_client):
        user_token = _register_and_login(flask_client, "l@test.com", "pass123", "L")
        pred_r = flask_client.post("/api/predict", json=LOAN_PAYLOAD,
                                   headers={"Authorization": f"Bearer {user_token}"})
        app_id = pred_r.get_json()["application_id"]

        r2 = flask_client.post("/api/login",
                               json={"email": "admin@bank.com", "password": "admin123"})
        admin_token = r2.get_json()["token"]
        flask_client.post(f"/api/admin/decision/{app_id}",
                          json={"status": "Rejected", "note": "Too risky"},
                          headers={"Authorization": f"Bearer {admin_token}"})

        app_r = flask_client.get(f"/api/application/{app_id}",
                                 headers={"Authorization": f"Bearer {user_token}"})
        assert app_r.get_json()["status"] == "Rejected"

    def test_invalid_admin_decision_status(self, flask_client):
        r = flask_client.post("/api/login",
                              json={"email": "admin@bank.com", "password": "admin123"})
        admin_token = r.get_json()["token"]
        r2 = flask_client.post("/api/admin/decision/1",
                               json={"status": "INVALID", "note": ""},
                               headers={"Authorization": f"Bearer {admin_token}"})
        assert r2.status_code == 400

    def test_user_cannot_view_other_users_application(self, flask_client):
        # User A submits
        token_a = _register_and_login(flask_client, "a@test.com", "pass123", "A")
        pred_r  = flask_client.post("/api/predict", json=LOAN_PAYLOAD,
                                    headers={"Authorization": f"Bearer {token_a}"})
        app_id  = pred_r.get_json()["application_id"]

        # User B tries to access User A's application
        token_b = _register_and_login(flask_client, "b@test.com", "pass123", "B")
        r = flask_client.get(f"/api/application/{app_id}",
                             headers={"Authorization": f"Bearer {token_b}"})
        assert r.status_code == 403

    def test_prediction_missing_field_returns_400(self, flask_client):
        token = _register_and_login(flask_client, "m@test.com", "pass123", "M")
        bad_payload = {k: v for k, v in LOAN_PAYLOAD.items() if k != "monthly_income"}
        r = flask_client.post("/api/predict", json=bad_payload,
                              headers={"Authorization": f"Bearer {token}"})
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-q"])
