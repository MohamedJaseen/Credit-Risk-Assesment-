"""
run.py — Convenience launcher.

Usage:
    python run.py backend     # start Flask API on :5000
    python run.py frontend    # start Streamlit on :8501
    python run.py train       # train ML models
    python run.py test        # run test suite
    python run.py analyze     # analyze dataset
    python run.py all         # start backend + frontend together
"""

import sys
import os
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))

# Ensure required directories exist
for folder in ["reports", "uploads", "models", "data"]:
    os.makedirs(os.path.join(BASE, folder), exist_ok=True)


def run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=BASE, **kwargs)


def backend():
    port = os.environ.get("PORT", "5000")
    print(f"🚀  Starting Flask backend on 0.0.0.0:{port} …")
    if os.name != "nt":  # Linux/macOS
        try:
            subprocess.run(["gunicorn", "--version"], capture_output=True)
            run(["gunicorn", "--bind", f"0.0.0.0:{port}", "backend.app:app"])
            return
        except Exception:
            pass
    # Fallback to dev server
    run([sys.executable, "backend/app.py"])


def frontend():
    port = os.environ.get("PORT", "8501")
    print(f"🎨  Starting Streamlit frontend on http://0.0.0.0:{port} …")
    run([sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", port, "--server.address", "0.0.0.0", "--server.headless", "true"])


def train():
    print("🤖  Training ML models …")
    run([sys.executable, "train_model.py"])


def test():
    print("🧪  Running tests …")
    run([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])


def analyze():
    print("📊  Analyzing dataset …")
    run([sys.executable, "analyze_dataset.py"])


def all_services():
    """Start backend and frontend in parallel."""
    import threading
    t1 = threading.Thread(target=backend, daemon=True)
    t2 = threading.Thread(target=frontend, daemon=True)
    t1.start()
    import time; time.sleep(2)
    t2.start()
    print("\n✅  Both services started.")
    print("   Backend  → http://0.0.0.0:5000")
    print("   Frontend → http://0.0.0.0:8501")
    print("   Press Ctrl+C to stop.\n")
    try:
        t1.join(); t2.join()
    except KeyboardInterrupt:
        print("\nStopped.")


def serve_static():
    port = os.environ.get("PORT", "8000")
    print(f"🌐  Serving Static Site on http://0.0.0.0:{port} …")
    static_dir = os.path.join(BASE, "static_site")
    run([sys.executable, "-m", "http.server", port, "--directory", static_dir])


if __name__ == "__main__":
    commands = {
        "backend":  backend,
        "frontend": frontend,
        "static":   serve_static,
        "train":    train,
        "test":     test,
        "analyze":  analyze,
        "all":      all_services,
    }
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    fn  = commands.get(cmd)
    if fn:
        fn()
    else:
        print(f"Unknown command '{cmd}'. Choose from: {list(commands.keys())}")

