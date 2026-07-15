import os


DEFAULT_API_BASE = "http://127.0.0.1:5000"


def get_api_base():
    """Return the backend API base URL for local development or deployment."""
    for env_name in ("API_URL", "BACKEND_URL"):
        value = os.environ.get(env_name, "").strip()
        if value:
            return value.rstrip("/")
    return DEFAULT_API_BASE
