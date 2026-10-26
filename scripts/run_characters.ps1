param(
    [ValidateSet('Test','Prepare','Train','Evaluate','Demo')]
    [string]$Action = 'Demo'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
# Use the environment installed for this checkout.
$characterPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $characterPython)) {
    throw 'Create .venv in this project and install requirements.txt first. See docs/guides/character-models.md.'
}
switch ($Action) {
    'Test' { & $characterPython -m unittest discover -s tests -v }
    'Prepare' { & $characterPython -m hnrs_chars.prepare --download }
    'Train' { & $characterPython -m hnrs_chars.train }
    'Evaluate' { & $characterPython -m hnrs_chars.evaluate }
    'Demo' {
        $selection = Get-Content -LiteralPath 'experiments/emnist_byclass/run_01/selection.json' -Raw | ConvertFrom-Json
        $checkpoint = 'experiments/emnist_byclass/run_01/' + $selection.selected_model + '.pt'
        $crops = @(0..6 | ForEach-Object { 'evidence/emnist/evaluation_run_01/example_crops/character_' + $_ + '.png' })
        & $characterPython -m hnrs_chars.predict --checkpoint $checkpoint --crops $crops
    }
}
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
