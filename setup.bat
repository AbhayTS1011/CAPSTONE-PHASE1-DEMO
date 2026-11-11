@echo off
REM One-time setup: Python virtual environment + Node packages + production build of the frontend.
setlocal
cd /d "%~dp0"

set "PYCMD="
for %%V in (3.10 3.11 3.12) do (
  if not defined PYCMD (
    py -%%V -c "import sys" >nul 2>&1 && set "PYCMD=py -%%V"
  )
)
if not defined PYCMD (
  python -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,12)" >nul 2>&1 && set "PYCMD=python"
)
if not defined PYCMD (
  echo [setup] Python 3.10, 3.11 or 3.12 is required. Install it from https://www.python.org/downloads/windows/
  exit /b 1
)
where npm >nul 2>&1 || (echo [setup] Node.js 20+ with npm is required. Install it from https://nodejs.org/ & exit /b 1)

echo [setup] Using %PYCMD%
if not exist backend\.venv\Scripts\python.exe (
  %PYCMD% -m venv backend\.venv || exit /b 1
)
backend\.venv\Scripts\python -m pip install --upgrade pip || exit /b 1
backend\.venv\Scripts\python -m pip install -r backend\requirements.txt || exit /b 1

cd frontend
call npm ci --no-fund --no-audit || exit /b 1
call npm run build || exit /b 1
cd ..
echo.
echo [setup] Done. Run start_app.bat and open http://127.0.0.1:8000
endlocal
