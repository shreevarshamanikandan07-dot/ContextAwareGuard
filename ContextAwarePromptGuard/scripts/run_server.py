"""
ContextAwarePromptGuard Application Server Launcher.
Starts the FastAPI server with Uvicorn hosting the REST API and the interactive web dashboard.
Run with:
    python ContextAwarePromptGuard/scripts/run_server.py
Access dashboard in your browser at:
    http://127.0.0.1:8000
"""

import os
import sys
import uvicorn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.join(BASE_DIR, "backend", "app")
sys.path.insert(0, APP_DIR)

if __name__ == "__main__":
    print("=" * 65)
    print("STARTING CONTEXT-AWARE PROMPT GUARD DASHBOARD SERVER")
    print("=" * 65)
    print("  Host: http://127.0.0.1:8000")
    print("  REST API docs: http://127.0.0.1:8000/docs")
    print("  Hardware: CPU-optimized PyTorch execution")
    print("  Press CTRL+C to stop.")
    print("=" * 65)
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, app_dir=APP_DIR)
