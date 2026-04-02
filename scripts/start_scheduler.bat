:: Starts scheduler.py as a detached background process (no console window).
:: Writes the process PID to logs\scheduler.pid so stop_scheduler.bat can kill it later.
:: Prevents duplicate instances by checking for an existing PID file before launching.
@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0.."

if not exist logs mkdir logs

:: ── Check if already running via PID file ────────────────────────────────
if exist logs\scheduler.pid (
    for /f "usebackq delims=" %%P in ("logs\scheduler.pid") do set EXISTING_PID=%%P
    if not "!EXISTING_PID!"=="" (
        tasklist /FI "PID eq !EXISTING_PID!" /FO CSV 2>nul | find "!EXISTING_PID!" >nul 2>&1
        if not errorlevel 1 (
            echo Scheduler is already running ^(PID !EXISTING_PID!^).
            echo Use stop_scheduler.bat to stop it first.
            exit /b 1
        )
        echo Stale PID file found ^(process !EXISTING_PID! not running^). Removing...
    )
    del logs\scheduler.pid
)

:: ── Launch as a fully detached process ───────────────────────────────────
:: DETACHED_PROCESS (0x8) + CREATE_NEW_PROCESS_GROUP (0x200) prevents the
:: parent cmd.exe from sending Ctrl+C to pythonw when this batch exits.
echo Starting Hedge Fund Scheduler...
echo Output will be written to logs\scheduler.log

python -c "import subprocess; f=subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP; p=subprocess.Popen(['pythonw','scheduler.py'],creationflags=f); open('logs/scheduler.pid','w').write(str(p.pid)); print('Scheduler started (PID '+str(p.pid)+')')"

if errorlevel 1 (
    echo ERROR: Failed to start scheduler. Make sure Python is in your PATH.
    exit /b 1
)

echo.
echo   scripts\stop_scheduler.bat         Stop the scheduler
echo   scripts\scheduler_status.bat       Check status and next run times
echo   python main.py --live-status        Show live data accumulation progress

endlocal
