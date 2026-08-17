"""
backend/auth.py
===============
Authentication: password hashing, JWT creation/validation, RBAC helpers.

Roles
-----
  user  — normal customer, can submit applications and view own data
  admin — bank officer, can view all data, approve/reject, train models
"""

import bcrypt
import jwt
import os
from datetime import datetime, timedelta
from typing import Optional
from functools import wraps
from flask import request, jsonify, g

from backend.database import create_user, get_user_by_email

SECRET    = os.environ.get("JWT_SECRET", "cris_college_project_2024")
ALGORITHM = "HS256"
EXP_HOURS = 8


# ─── Password helpers ───────────────────────────────────────────────────────

def hash_pw(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_pw(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ─── JWT helpers ────────────────────────────────────────────────────────────

def make_token(user_id: int, email: str, role: str, name: str) -> str:
    payload = {
        "sub":   str(user_id),
        "email": email,
        "role":  role,
        "name":  name,
        "exp":   datetime.utcnow() + timedelta(hours=EXP_HOURS),
    }
    return jwt.encode(payload, SECRET, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, SECRET, algorithms=[ALGORITHM])
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
        return None


def get_token_from_request() -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth.split(" ", 1)[1]
    return None


# ─── Flask decorators for RBAC ──────────────────────────────────────────────

def login_required(f):
    """Decorator: any authenticated user."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return f(*args, **kwargs)
        token = get_token_from_request()
        if not token:
            return jsonify({"error": "Authentication required. Please log in."}), 401
        payload = decode_token(token)
        if not payload:
            return jsonify({"error": "Session expired. Please log in again."}), 401
        user = get_user_by_email(payload.get("email"))
        if user:
            payload["sub"] = str(user["id"])
        g.user = payload   # attach user info to Flask's g object
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    """Decorator: admin role only."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        if request.method == "OPTIONS":
            return f(*args, **kwargs)
        token = get_token_from_request()
        if not token:
            return jsonify({"error": "Authentication required."}), 401
        payload = decode_token(token)
        if not payload:
            return jsonify({"error": "Session expired. Please log in again."}), 401
        if payload.get("role") != "admin":
            return jsonify({"error": "Access denied. Admin privileges required."}), 403
        user = get_user_by_email(payload.get("email"))
        if user:
            payload["sub"] = str(user["id"])
        g.user = payload
        return f(*args, **kwargs)
    return wrapper



# ─── Auth service functions ─────────────────────────────────────────────────

def register_service(name: str, email: str, password: str) -> dict:
    if len(password) < 6:
        return {"ok": False, "msg": "Password must be at least 6 characters."}
    if not email or "@" not in email:
        return {"ok": False, "msg": "Please enter a valid email address."}
    pw_hash = hash_pw(password)
    return create_user(name, email, pw_hash, role="user")


def login_service(email: str, password: str) -> dict:
    user = get_user_by_email(email)
    if not user:
        return {"ok": False, "msg": "No account found with this email."}
    if not verify_pw(password, user["password_hash"]):
        return {"ok": False, "msg": "Incorrect password. Please try again."}
    token = make_token(user["id"], user["email"], user["role"], user["name"])
    return {
        "ok": True,
        "token": token,
        "user": {
            "id":    user["id"],
            "name":  user["name"],
            "email": user["email"],
            "role":  user["role"],
        },
    }
