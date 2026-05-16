$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CondaExe = "D:\Anaconda\Scripts\conda.exe"
$TorchPython = "D:\Anaconda\envs\torch\python.exe"

if (-not (Test-Path $CondaExe)) {
    throw "conda.exe not found: $CondaExe"
}

Set-Location $ProjectRoot
$env:WAV2LIP_DIR = Join-Path $ProjectRoot "Wav2Lip"
$env:WAV2LIP_PYTHON = $TorchPython
& $CondaExe run -n torch python .\web_app.py
