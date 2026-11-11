@echo off
REM Starts the local FastAPI backend on http://127.0.0.1:8000 (also serves the built frontend).
cd /d "%~dp0backend"
if not exist .venv\Scripts\python.exe (
  echo [start_backend] Run setup.bat first.
  exit /b 1
)
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
