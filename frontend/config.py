import os


DEFAULT_API_BASE = "http://127.0.0.1:5000"
RENDER_API_BASE = "https://credit-risk-api.onrender.com"


def get_api_base():
    """Return the backend API base URL for local development or deployment."""
    for env_name in ("API_URL", "BACKEND_URL"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value.rstrip("/")

    if os.environ.get("RENDER") == "true" or os.environ.get("RENDER") == "True":
        return RENDER_API_BASE

    return DEFAULT_API_BASE
