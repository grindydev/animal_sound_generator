"""
start.py — Phase 6: One-command launcher
=========================================

Usage:
    python start.py

This starts the FastAPI server and opens the browser.
"""
import subprocess
import sys
import webbrowser
import time
import os
from pathlib import Path

os.chdir(Path(__file__).parent)

# Start server
print("🐾 Starting Animal Sound Generator...")
print("   Server: http://localhost:8000")
print("   Press Ctrl+C to stop\n")

server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=str(Path(__file__).parent),
)

# Open browser after a short delay
time.sleep(2)
webbrowser.open("http://localhost:8000")

try:
    server_proc.wait()
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    server_proc.terminate()
