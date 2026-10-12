param(
    [ValidateSet('Demo','Compare','Test')][string]$Action = 'Demo',
    [string]$Image = 'data/own_strings/development/Manula/Manula_01_text-Ab12.jpg',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$stringPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $stringPython)) {
    $download = Join-Path (Split-Path -Parent $PSScriptRoot) 'assignment options\COS30018-Intelligent-Systems-project-haresh-ml-baseline\COS30018-Intelligent-Systems-project-haresh-ml-baseline'
    $stringPython = Join-Path $download '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $stringPython)) { throw 'Create .venv and install requirements.txt first.' }
if (-not $Output) { $Output = 'evidence/character_segmentation/' + $Action.ToLower() + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff') }
switch ($Action) {
    'Demo' { & $stringPython -m hnrs_chars.string_predict --image $Image --output $Output }
    'Compare' { & $stringPython -m hnrs_chars.string_experiment --output $Output }
    'Test' { & $stringPython -m unittest discover -s tests -v }
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
