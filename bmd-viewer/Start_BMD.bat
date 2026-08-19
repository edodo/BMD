@echo off
title BMD Viewer - Starting
cd /d "%~dp0"

echo ============================================
echo   Starting BMD Viewer...
echo ============================================
echo.

REM ---- 1) Make sure Docker Desktop is running ----
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready

echo Docker Desktop is not running yet. Starting it now, please wait...
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"

set /a count=0
:wait_docker
timeout /t 3 >nul
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready
set /a count+=1
if %count% GEQ 40 (
    echo.
    echo [ERROR] Docker Desktop did not become ready within 5 minutes.
    echo Please start Docker Desktop manually from the desktop/start menu,
    echo wait until the whale icon shows it is fully running, then run this file again.
    echo.
    pause
    exit /b 1
)
goto wait_docker

:docker_ready
echo Docker is ready.
echo.

REM ---- 2) Start the containers (first run will build automatically, may take a while) ----
echo Preparing the application...
echo (If this is the first time, it may take 5-10 minutes. Please keep this window open.)
echo.
docker compose up -d
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to start. Please check the messages above.
    echo If the problem continues, take a screenshot of this window and ask for help.
    echo.
    pause
    exit /b 1
)

echo.
echo Waiting for the server to become ready...
set /a wcount=0
:wait_server
curl --fail -s -o nul http://localhost >nul 2>&1
if not errorlevel 1 goto server_ready
timeout /t 2 >nul
set /a wcount+=1
if %wcount% GEQ 60 goto server_ready
goto wait_server

:server_ready
echo.
echo ============================================
echo   BMD Viewer is ready!
echo   Your browser will open automatically.
echo   (If the page looks empty, wait a few seconds and refresh)
echo ============================================
echo.
start http://localhost
timeout /t 2 >nul
echo You can close this window now.
echo Press any key to close.
pause >nul
