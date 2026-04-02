:: Stops the running scheduler by reading the PID from logs\scheduler.pid and
:: terminating the process with taskkill. Cleans up the PID file on exit.
:: Use after start_scheduler.bat.
@echo off
setlocal
cd /d "%~dp0.."

if not exist logs\scheduler.pid (
    echo Scheduler PID file not found.
    echo The scheduler may not be running, or was started without start_scheduler.bat.
    echo.
    echo To stop any pythonw process manually:
    echo   taskkill /IM pythonw.exe /F
    exit /b 1
)

set /p PID=<logs\scheduler.pid

if "%PID%"=="" (
    echo PID file is empty. Removing and exiting.
    del logs\scheduler.pid
    exit /b 1
)

echo Stopping scheduler (PID %PID%)...
taskkill /PID %PID% /F

if not errorlevel 1 (
    echo Scheduler stopped successfully.
) else (
    echo Process %PID% could not be found — it may have already exited.
)

del logs\scheduler.pid
echo PID file removed.
endlocal
