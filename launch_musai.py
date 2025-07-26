import os
import signal
import subprocess
import time
from pathlib import Path

import requests

# 🎛️ Environment flags for Playwright
# Note: PWDEBUG=1 enables manual inspector mode - disable for automated runs
# os.environ["PWDEBUG"] = "1"  # Commented out to prevent manual 'play' requirement
os.environ["DEBUG"] = "pw:api"  # Keep API debugging for logs

# 📁 Setup paths
PROJECT_DIR = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_DIR / ".venv/bin/python"
MAIN_SCRIPT = PROJECT_DIR / "main.py"
CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
DEBUG_PORT = 9222
DEBUG_USER_DATA_DIR = "/tmp/playwright"


def kill_existing_chrome():
    print("🔪 Killing any existing Chrome processes...")
    subprocess.call(
        ["pkill", "-f", "Google Chrome"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def start_debug_chrome():
    print("🚀 Starting Chrome in remote debug mode...")
    return subprocess.Popen(
        [
            CHROME_PATH,
            f"--remote-debugging-port={DEBUG_PORT}",
            f"--user-data-dir={DEBUG_USER_DATA_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_for_debug_endpoint(timeout=15):
    print(f"⏳ Waiting for Chrome DevTools on http://localhost:{DEBUG_PORT}/json ...")
    for _ in range(timeout * 2):
        try:
            res = requests.get(f"http://localhost:{DEBUG_PORT}/json")
            if res.status_code == 200:
                print("✅ Chrome DevTools is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    print("❌ Timeout: DevTools never became ready.")
    return False


def run_musai_agent():
    if not VENV_PYTHON.exists():
        print(f"❗ Virtualenv Python not found: {VENV_PYTHON}")
        return
    if not MAIN_SCRIPT.exists():
        print(f"❗ Main script not found: {MAIN_SCRIPT}")
        return

    print(f"🧠 Running Musai agent with Python: {VENV_PYTHON}")
    result = subprocess.run(
        [str(VENV_PYTHON), str(MAIN_SCRIPT)],
        cwd=str(PROJECT_DIR),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    print(f"✅ Agent exited with code {result.returncode}")


if __name__ == "__main__":
    kill_existing_chrome()
    chrome_proc = start_debug_chrome()
    try:
        if wait_for_debug_endpoint():
            run_musai_agent()
        else:
            print("🛑 Aborting launch: Chrome failed to open in debug mode.")
    finally:
        print("🧹 Cleaning up Chrome...")
        try:
            os.kill(chrome_proc.pid, signal.SIGTERM)
        except Exception as e:
            print(f"⚠️ Failed to kill Chrome: {e}")
