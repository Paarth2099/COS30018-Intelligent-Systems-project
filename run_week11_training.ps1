param(
    [ValidateSet('Check','Train')][string]$Action = 'Check',
    [ValidateSet('all','mnist','emnist')][string]$Task = 'all',
    [string]$Output = 'experiments/week11_gpu_run01'
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$trainingPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $trainingPython)) {
    throw 'Create this PC''s .venv and install CUDA-enabled PyTorch first.'
}
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONUTF8 = '1'
if ($Action -eq 'Check') {
    & $trainingPython -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print('Torch:', torch.__version__)"
    if ($LASTEXITCODE -ne 0) { throw 'GPU detection failed.' }
    & $trainingPython -m unittest discover -s tests -p test_extended_models.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Model contract tests failed.' }
    & $trainingPython -m unittest discover -s tests -p test_week11_training.py -v
    if ($LASTEXITCODE -ne 0) { throw 'Training tests failed.' }
    & $trainingPython -c "from train_models_week11 import data; print('Checking original training splits'); [print(task, len(data(task)[2]), 'training examples') for task in ('mnist','emnist')]"
    if ($LASTEXITCODE -ne 0) { throw 'Dataset/split checks failed. Copy data and evidence/emnist/preparation from the original project.' }
    $pilotName = 'pilot_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
    $pilotOutput = Join-Path $Output $pilotName
    & $trainingPython -u train_models_week11.py --pilot --device cuda --output $pilotOutput
} else {
    & $trainingPython -u train_models_week11.py --task $Task --device cuda --epochs 5 --batch-size 256 --output $Output
}
if ($LASTEXITCODE -ne 0) { throw 'Week 11 command failed. Preserve the output and inspect the error before restarting.' }
