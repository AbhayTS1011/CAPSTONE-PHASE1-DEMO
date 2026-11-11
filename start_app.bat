@echo off
REM Starts the backend in its own window and opens the application in the default browser.
cd /d "%~dp0"
if not exist frontend\dist\index.html (
  echo [start_app] The frontend has not been built. Run setup.bat first.
  exit /b 1
)
start "PSJ_01 backend" cmd /k "%~dp0start_backend.bat"
REM give uvicorn a few seconds to start
ping -n 5 127.0.0.1 >nul
start "" http://127.0.0.1:8000
echo The application is running at http://127.0.0.1:8000  (close the "PSJ_01 backend" window to stop it)
