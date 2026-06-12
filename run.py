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


def run(cmd, **kwargs):
    return subprocess.run(cmd, cwd=BASE, **kwargs)


def backend():
    print("🚀  Starting Flask backend on http://localhost:5000 …")
    run([sys.executable, "backend/app.py"])


def frontend():
    print("🎨  Starting Streamlit frontend on http://localhost:8501 …")
    run([sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", "8501", "--server.headless", "true"])


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
    print("   Backend  → http://localhost:5000")
    print("   Frontend → http://localhost:8501")
    print("   Press Ctrl+C to stop.\n")
    try:
        t1.join(); t2.join()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    commands = {
        "backend":  backend,
        "frontend": frontend,
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
