"""
backend/database.py
===================
Single source of truth for all database operations.

Tables
------
  users        — stores users with roles (user / admin)
  applications — single table with status column (pending/approved/rejected)
  decisions    — audit trail of every admin decision
"""

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime

# ─── Path setup ────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "data", "credit_risk.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


@contextmanager
def get_conn():
    """Thread-safe SQLite context manager."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Bootstrap ─────────────────────────────────────────────────────────────

def init_db():
    """Create all tables and seed the default admin account."""
    with get_conn() as conn:
        conn.executescript("""
        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            email         TEXT    UNIQUE NOT NULL,
            password_hash TEXT    NOT NULL,
            role          TEXT    NOT NULL DEFAULT 'user',
            created_at    TEXT    DEFAULT (datetime('now'))
        );

        -- Applications table — one table, status column drives the workflow
        CREATE TABLE IF NOT EXISTS applications (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER NOT NULL,
            user_name        TEXT,
            user_email       TEXT,

            -- raw user inputs (stored as JSON string)
            input_json       TEXT,

            -- ML outputs
            credit_score     INTEGER,
            score_label      TEXT,
            probability      REAL,
            risk_category    TEXT,     -- LOW / MEDIUM / HIGH
            ml_recommendation TEXT,    -- "Recommend: Approve" or "Recommend: Reject"

            -- AI explanation
            reason_codes     TEXT,     -- comma-separated
            shap_explanation TEXT,
            recommendations  TEXT,     -- JSON list

            -- Workflow columns  ← KEY IMPROVEMENT
            status           TEXT    NOT NULL DEFAULT 'Pending',
            -- status values: Pending | Approved | Rejected

            -- Admin review columns
            reviewed_by      TEXT,    -- admin email
            review_note      TEXT,
            reviewed_at      TEXT,

            created_at       TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        -- Decisions audit table — every approve/reject action by admin
        CREATE TABLE IF NOT EXISTS decisions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            application_id INTEGER NOT NULL,
            admin_email    TEXT,
            old_status     TEXT,
            new_status     TEXT,
            note           TEXT,
            decided_at     TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(application_id) REFERENCES applications(id)
        );
        """)

    _seed_admin()


def _seed_admin():
    """Create a default admin account if none exists."""
    import bcrypt
    with get_conn() as conn:
        row = conn.execute("SELECT id FROM users WHERE role='admin' LIMIT 1").fetchone()
        if not row:
            pw = bcrypt.hashpw(b"admin123", bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (name, email, password_hash, role) VALUES (?,?,?,?)",
                ("Admin", "admin@bank.com", pw, "admin")
            )


# ─── User operations ────────────────────────────────────────────────────────

def create_user(name, email, password_hash, role="user"):
    with get_conn() as conn:
        existing = conn.execute("SELECT id FROM users WHERE email=?", (email,)).fetchone()
        if existing:
            return {"ok": False, "msg": "This email is already registered."}
        conn.execute(
            "INSERT INTO users (name, email, password_hash, role) VALUES (?,?,?,?)",
            (name, email, password_hash, role)
        )
    return {"ok": True, "msg": "Account created successfully."}


def get_user_by_email(email):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None


def get_all_users():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, email, role, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


# ─── Application operations ─────────────────────────────────────────────────

def save_application(data: dict) -> int:
    """Save a new application and return its ID."""
    with get_conn() as conn:
        cur = conn.execute("""
            INSERT INTO applications
                (user_id, user_name, user_email, input_json,
                 credit_score, score_label, probability, risk_category,
                 ml_recommendation, reason_codes, shap_explanation, recommendations,
                 status)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'Pending')
        """, (
            data["user_id"], data["user_name"], data["user_email"],
            data["input_json"],
            data["credit_score"], data["score_label"],
            data["probability"], data["risk_category"],
            data["ml_recommendation"],
            data["reason_codes"], data["shap_explanation"],
            data["recommendations"],
        ))
        return cur.lastrowid


def get_application(app_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM applications WHERE id=?", (app_id,)).fetchone()
        return dict(row) if row else None


def get_user_applications(user_id: int):
    """All applications belonging to a specific user."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM applications WHERE user_id=? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_all_applications(status_filter=None, risk_filter=None):
    """All applications — admin view, with optional filters."""
    query  = "SELECT * FROM applications WHERE 1=1"
    params = []
    if status_filter and status_filter != "All":
        query += " AND status=?";       params.append(status_filter)
    if risk_filter and risk_filter != "All":
        query += " AND risk_category=?"; params.append(risk_filter)
    query += " ORDER BY created_at DESC"
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


def admin_decide(app_id: int, new_status: str, admin_email: str, note: str):
    """Admin approves or rejects an application and logs the decision."""
    if new_status not in ("Approved", "Rejected", "Pending"):
        return False
    with get_conn() as conn:
        old = conn.execute(
            "SELECT status FROM applications WHERE id=?", (app_id,)
        ).fetchone()
        if not old:
            return False
        old_status = old["status"]
        conn.execute("""
            UPDATE applications
               SET status=?, reviewed_by=?, review_note=?, reviewed_at=datetime('now')
             WHERE id=?
        """, (new_status, admin_email, note, app_id))
        # Log to decisions table
        conn.execute("""
            INSERT INTO decisions (application_id, admin_email, old_status, new_status, note)
            VALUES (?,?,?,?,?)
        """, (app_id, admin_email, old_status, new_status, note))
    return True


def get_decision_history(app_id: int):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM decisions WHERE application_id=? ORDER BY decided_at DESC",
            (app_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# ─── Admin analytics ────────────────────────────────────────────────────────

def get_dashboard_stats():
    with get_conn() as conn:
        total     = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
        pending   = conn.execute("SELECT COUNT(*) FROM applications WHERE status='Pending'").fetchone()[0]
        approved  = conn.execute("SELECT COUNT(*) FROM applications WHERE status='Approved'").fetchone()[0]
        rejected  = conn.execute("SELECT COUNT(*) FROM applications WHERE status='Rejected'").fetchone()[0]
        high_risk = conn.execute("SELECT COUNT(*) FROM applications WHERE risk_category='HIGH'").fetchone()[0]
        med_risk  = conn.execute("SELECT COUNT(*) FROM applications WHERE risk_category='MEDIUM'").fetchone()[0]
        low_risk  = conn.execute("SELECT COUNT(*) FROM applications WHERE risk_category='LOW'").fetchone()[0]
        avg_score = conn.execute("SELECT AVG(credit_score) FROM applications").fetchone()[0]
        total_users = conn.execute("SELECT COUNT(*) FROM users WHERE role='user'").fetchone()[0]

        # Monthly trend (last 6 months)
        monthly = conn.execute("""
            SELECT strftime('%Y-%m', created_at) AS month,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='Approved' THEN 1 ELSE 0 END) AS approved,
                   SUM(CASE WHEN status='Rejected' THEN 1 ELSE 0 END) AS rejected
              FROM applications
             WHERE created_at >= datetime('now','-6 months')
             GROUP BY month ORDER BY month
        """).fetchall()

    return {
        "total": total, "pending": pending,
        "approved": approved, "rejected": rejected,
        "high_risk": high_risk, "med_risk": med_risk, "low_risk": low_risk,
        "avg_credit_score": round(avg_score or 0, 1),
        "total_users": total_users,
        "monthly_trend": [dict(r) for r in monthly],
    }
