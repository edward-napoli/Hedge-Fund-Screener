:: Displays the scheduler status and next scheduled run times.
:: Runs: python main.py --scheduler-status
@echo off
cd /d "%~dp0.."
python main.py --scheduler-status
