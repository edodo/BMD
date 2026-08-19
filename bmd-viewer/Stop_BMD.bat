@echo off
title BMD Viewer - Stopping
cd /d "%~dp0"

echo ============================================
echo   Stopping BMD Viewer...
echo ============================================
echo.

docker compose down
if errorlevel 1 (
    echo.
    echo [ERROR] Something went wrong while stopping.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   BMD Viewer has been stopped.
echo   Your saved patient data is kept safe and
echo   will still be there next time you start it.
echo ============================================
echo.
echo Press any key to close this window.
pause >nul
