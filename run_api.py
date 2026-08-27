"""Launcher: `python run_api.py` starts the FastAPI backend (src/api.py) with
uvicorn, without needing to remember the module path / uvicorn CLI flags.
Mirrors run_demo.py's role for the Streamlit dashboard.

No --reload here on purpose: uvicorn's default file watcher scans the whole
working directory, and this repo's cwd contains dataset/ (multi-GB of
SemanticPOSS .bin/.label files) and .venv/ -- watching those makes the
reloader's startup scan take a very long time. For active backend
development, run uvicorn directly with a scoped watch instead:
    uvicorn src.api:app --reload --reload-dir src --port 8000
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "uvicorn", "src.api:app",
        "--host", "127.0.0.1", "--port", "8000",
    ])
