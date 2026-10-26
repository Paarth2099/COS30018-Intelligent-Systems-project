param(
    [ValidateSet('Demo','Compare','Test')][string]$Action = 'Demo',
    [string]$Image = 'data/own_characters/development/Paarth/Paarth_02_lower-i.jpg',
    [string]$Output = ''
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$photoPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $photoPython)) {
    throw 'Create .venv and install requirements.txt. See docs/guides/character-preprocessing.md.'
}
if (-not $Output) {
    $Output = 'evidence/character_preprocessing/' + $Action.ToLower() + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
}
switch ($Action) {
    'Demo' { & $photoPython -m hnrs_chars.photo_predict --image $Image --output $Output }
    'Compare' { & $photoPython -m hnrs_chars.photo_experiment --output $Output }
    'Test' { & $photoPython -m unittest discover -s tests -v }
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
