$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CondaRoot = "D:\Anaconda"
$CondaExe = Join-Path $CondaRoot "Scripts\conda.exe"
$GitExe = "git"
$EnvName = "torch"
$RepoDir = Join-Path $ProjectRoot "ditto-talkinghead"
$CheckpointDir = Join-Path $RepoDir "checkpoints"
$HfMirrorEndpoint = "https://hf-mirror.com"
$HfOfficialEndpoint = "https://huggingface.co"
$HfCheckpointPath = "digital-avatar/ditto-talkinghead"
$HfCheckpointRepoMirror = "$HfMirrorEndpoint/$HfCheckpointPath"
$HfCheckpointRepoOfficial = "$HfOfficialEndpoint/$HfCheckpointPath"

function Invoke-CondaRun {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    & $CondaExe run -n $EnvName @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed: $($Args -join ' ')"
    }
}

if (-not (Test-Path $CondaExe)) {
    throw "conda.exe not found: $CondaExe"
}

Write-Host "==> Verify torch environment"
Invoke-CondaRun -Args @("python", "-V")

Write-Host "==> Set Hugging Face mirror endpoint"
$env:HF_ENDPOINT = $HfMirrorEndpoint
Write-Host "HF_ENDPOINT=$env:HF_ENDPOINT"

Write-Host "==> Prepare Ditto TalkingHead repository"
if (-not (Test-Path $RepoDir)) {
    & $GitExe clone https://github.com/antgroup/ditto-talkinghead $RepoDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone ditto-talkinghead repository."
    }
} else {
    Write-Host "Repository already exists, skip clone: $RepoDir"
}

Write-Host "==> Prepare Git LFS"
& $GitExe lfs install
if ($LASTEXITCODE -ne 0) {
    throw "git lfs install failed."
}

Write-Host "==> Upgrade pip tooling"
Invoke-CondaRun -Args @("python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

$RequirementsTxt = Join-Path $RepoDir "requirements.txt"
$PyProject = Join-Path $RepoDir "pyproject.toml"
$SetupPy = Join-Path $RepoDir "setup.py"

Write-Host "==> Install Ditto dependencies"
if (Test-Path $RequirementsTxt) {
    Invoke-CondaRun -Args @("python", "-m", "pip", "install", "-r", $RequirementsTxt)
} elseif ((Test-Path $PyProject) -or (Test-Path $SetupPy)) {
    Invoke-CondaRun -Args @("python", "-m", "pip", "install", "-e", $RepoDir)
} else {
    Write-Warning "No requirements.txt, pyproject.toml, or setup.py found. Please check the repository manually."
}

Invoke-CondaRun -Args @("python", "-m", "pip", "install", "imageio-ffmpeg", "ffmpeg-python", "huggingface_hub")

Write-Host "==> Download public checkpoints"
if (-not (Test-Path $CheckpointDir)) {
    Write-Host "Try mirror first: $HfCheckpointRepoMirror"
    & $GitExe clone $HfCheckpointRepoMirror $CheckpointDir
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Mirror clone failed. Retry with official Hugging Face endpoint."
        if (Test-Path $CheckpointDir) {
            Remove-Item -LiteralPath $CheckpointDir -Recurse -Force
        }
        & $GitExe clone $HfCheckpointRepoOfficial $CheckpointDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to clone checkpoints from both mirror and official Hugging Face."
        }
    }
} else {
    Write-Host "Checkpoints already exist, skip clone: $CheckpointDir"
}

Write-Host "==> Validate key files"
$InferencePy = Join-Path $RepoDir "inference.py"
if (-not (Test-Path $InferencePy)) {
    throw "inference.py not found: $InferencePy"
}

$CfgCandidates = @(
    (Join-Path $CheckpointDir "ditto_cfg\v0.4_hubert_cfg_pytorch.pkl"),
    (Join-Path $CheckpointDir "ditto_cfg\v0.4_hubert_cfg_trt.pkl")
)

$ModelCandidates = @(
    (Join-Path $CheckpointDir "ditto_pytorch"),
    (Join-Path $CheckpointDir "ditto_trt_Ampere_Plus"),
    (Join-Path $CheckpointDir "ditto_trt_custom")
)

$CfgFound = $CfgCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
$ModelFound = $ModelCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $CfgFound) {
    throw "No Ditto config file found under checkpoints."
}

if (-not $ModelFound) {
    throw "No Ditto model directory found under checkpoints."
}

Write-Host ""
Write-Host "Installation completed."
Write-Host "Repo:        $RepoDir"
Write-Host "Checkpoints: $CheckpointDir"
Write-Host "CFG:         $CfgFound"
Write-Host "Model:       $ModelFound"
Write-Host ""
Write-Host "Start backend with:"
Write-Host "  D:\Anaconda\Scripts\conda.exe run -n torch python .\web_app.py"
