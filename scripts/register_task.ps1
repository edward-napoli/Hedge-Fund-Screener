# Registers scheduler.py as a Windows Task Scheduler task (HedgeFundScheduler) that
# auto-starts at logon using pythonw.exe (no console window). Restarts up to 3 times
# on failure with a 5-minute interval. Called by setup_windows_task.bat, or run directly:
#   powershell -ExecutionPolicy Bypass -File scripts\register_task.ps1

$ErrorActionPreference = "Stop"

$pythonw = "C:\Users\enapoli26\AppData\Local\Programs\Python\Python314\pythonw.exe"
$script  = "C:\Users\enapoli26\Hedge Fund Project\scheduler.py"
$workdir = "C:\Users\enapoli26\Hedge Fund Project"
$userId  = "REGIS\enapoli26"

# Write task XML — avoids all quoting and COM/CIM permission issues
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>Hedge Fund Stock Screener - fires twice daily on weekdays</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
      <UserId>$userId</UserId>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>$userId</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <RestartOnFailure>
      <Interval>PT5M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>$pythonw</Command>
      <Arguments>"$script"</Arguments>
      <WorkingDirectory>$workdir</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"@

$xmlPath = Join-Path $env:TEMP "HedgeFundScheduler.xml"
$xml | Out-File -FilePath $xmlPath -Encoding Unicode

$result = schtasks /Create /TN "HedgeFundScheduler" /XML $xmlPath /F 2>&1
Remove-Item $xmlPath -ErrorAction SilentlyContinue

if ($LASTEXITCODE -ne 0) {
    Write-Host "schtasks output: $result"
    throw "schtasks /Create failed (exit $LASTEXITCODE). Try running as Administrator."
}

Write-Host "Task registered successfully."
Write-Host "  Executable : $pythonw"
Write-Host "  Script     : $script"
Write-Host "  Trigger    : At logon ($userId)"
Write-Host ""
Write-Host "To run immediately : Start-Process schtasks -ArgumentList '/Run /TN HedgeFundScheduler'"
Write-Host "To disable         : schtasks /Change /TN HedgeFundScheduler /DISABLE"
Write-Host "To remove          : schtasks /Delete /TN HedgeFundScheduler /F"
