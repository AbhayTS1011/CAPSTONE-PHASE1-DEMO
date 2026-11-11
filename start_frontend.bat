@echo off
REM Development only: Vite dev server with hot reload on http://127.0.0.1:5173 (needs start_backend.bat running).
cd /d "%~dp0frontend"
if not exist node_modules (
  echo [start_frontend] Run setup.bat first.
  exit /b 1
)
call npm run dev
