"""Launcher: `python run_demo.py` starts the Streamlit dashboard without
needing to remember the streamlit CLI invocation."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    app_path = Path(__file__).resolve().parent / "src" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
