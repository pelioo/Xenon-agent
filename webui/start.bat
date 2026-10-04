@echo off
cd /d %~dp0..

set "PORT=8000"
if defined XENON_WEBUI_PORT set "PORT=%XENON_WEBUI_PORT%"

rem 若 WebUI 已在运行，直接提示，避免重复启动后闪退
curl -s -m 2 "http://127.0.0.1:%PORT%/health" 2>NUL | findstr /C:"ok" >NUL
if not errorlevel 1 (
    echo.
    echo [Xenon WebUI] 服务已在运行：http://127.0.0.1:%PORT%/
    echo 无需重复启动；如需重启，请先关闭现有实例。
    echo.
    pause
    exit /b 0
)

venv\Scripts\python.exe webui\main.py
pause
