@echo off
REM ===========================================================================
REM senz_hub.bat -- double-click launcher for the senz control hub.
REM
REM Finds the Python on your PATH and launches senz_hub.py with the matching
REM pythonw.exe (windowed, so no console window lingers). If Python is missing
REM it says so instead of silently doing nothing. For a completely flash-free
REM launch, double-click senz_hub.vbs instead (it runs this hidden).
REM ===========================================================================
setlocal
cd /d "%~dp0"

set "PYEXE="
for /f "delims=" %%p in ('where python 2^>nul') do (
    if not defined PYEXE set "PYEXE=%%p"
)

if not defined PYEXE (
    echo.
    echo   Could not find Python on your PATH.
    echo   Install Python 3, or add it to PATH, then try again.
    echo.
    pause
    exit /b 1
)

REM pythonw.exe (no console) lives next to python.exe.
for %%d in ("%PYEXE%") do set "PYW=%%~dpdpythonw.exe"

if exist "%PYW%" (
    start "" "%PYW%" senz_hub.py
) else (
    REM No pythonw found; fall back to python.exe (a console window will show).
    start "" "%PYEXE%" senz_hub.py
)
