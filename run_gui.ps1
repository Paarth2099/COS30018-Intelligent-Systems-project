param([switch]$Verify)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$guiPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $guiPython)) {
    $baselineFolder = Join-Path (Split-Path -Parent $PSScriptRoot) 'assignment options\COS30018-Intelligent-Systems-project-haresh-ml-baseline\COS30018-Intelligent-Systems-project-haresh-ml-baseline'
    $guiPython = Join-Path $baselineFolder '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $guiPython)) {
    throw 'Create .venv using Python 3.12 and install requirements.txt. See SANIRU_EXTENSION.md.'
}
if ($Verify) { & $guiPython tests/gui_extension_smoke.py }
else { & $guiPython app.py }
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
