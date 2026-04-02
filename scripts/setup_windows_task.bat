:: Registers the scheduler as a Windows Task Scheduler task (HedgeFundScheduler)
:: that auto-starts at logon. Calls register_task.ps1 via PowerShell.
:: Run once, as Administrator if prompted, to set up the startup task.
@echo off
cd /d "%~dp0.."
echo Registering HedgeFundScheduler as a Windows Task Scheduler task...
powershell -ExecutionPolicy Bypass -File "%~dp0register_task.ps1"
