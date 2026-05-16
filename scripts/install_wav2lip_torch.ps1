$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$CondaRoot = "D:\Anaconda"
$CondaExe = Join-Path $CondaRoot "Scripts\conda.exe"
$GitExe = "git"
$EnvName = "torch"
$RepoDir = Join-Path $ProjectRoot "Wav2Lip"
$CheckpointDir = Join-Path $RepoDir "checkpoints"
$ResultsDir = Join-Path $RepoDir "results"
$ModelPath = Join-Path $CheckpointDir "wav2lip_gan.pth"
$ManualDownloadsDir = Join-Path $ProjectRoot "downloads"
$ManualModelPath = Join-Path $ManualDownloadsDir "wav2lip_gan.pth"
$FaceDetDir = Join-Path $RepoDir "face_detection\detection\sfd"
$FaceDetPath = Join-Path $FaceDetDir "s3fd.pth"
$Wav2LipDriveUrl = "https://drive.google.com/uc?id=1jQOJInh8cDj2mrbUgcQxhCc7rpAgyV1-"
$S3fdUrl = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"

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

Write-Host "==> Prepare Wav2Lip repository"
if (-not (Test-Path $RepoDir)) {
    & $GitExe clone https://github.com/Rudrabha/Wav2Lip $RepoDir
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to clone Wav2Lip repository."
    }
} else {
    Write-Host "Repository already exists, skip clone: $RepoDir"
}

New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
New-Item -ItemType Directory -Force -Path $FaceDetDir | Out-Null
New-Item -ItemType Directory -Force -Path $ManualDownloadsDir | Out-Null

Write-Host "==> Install ffmpeg into torch environment"
& $CondaExe install -n $EnvName -y -c conda-forge ffmpeg
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install ffmpeg with conda."
}

Write-Host "==> Upgrade pip tooling"
Invoke-CondaRun -Args @("python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")

Write-Host "==> Install inference dependencies"
Invoke-CondaRun -Args @(
    "python", "-m", "pip", "install",
    "gdown",
    "imageio-ffmpeg",
    "ffmpeg-python",
    "numpy<2",
    "scipy<1.12",
    "librosa>=0.10,<0.11",
    "numba<0.61",
    "opencv-python<4.10",
    "opencv-contrib-python<4.10",
    "tqdm"
)

Write-Host "==> Download Wav2Lip GAN checkpoint"
if (-not (Test-Path $ModelPath)) {
    if (Test-Path $ManualModelPath) {
        Copy-Item -LiteralPath $ManualModelPath -Destination $ModelPath -Force
        Write-Host "Copied manual checkpoint from: $ManualModelPath"
    } else {
        try {
            Invoke-CondaRun -Args @("python", "-m", "gdown", $Wav2LipDriveUrl, "-O", $ModelPath)
        } catch {
            throw "Failed to download wav2lip_gan.pth from Google Drive. Please manually place the file at: $ManualModelPath and rerun this script."
        }
    }
} else {
    Write-Host "Checkpoint already exists: $ModelPath"
}

Write-Host "==> Download S3FD face detector checkpoint"
if (-not (Test-Path $FaceDetPath)) {
    Invoke-WebRequest -Uri $S3fdUrl -OutFile $FaceDetPath
} else {
    Write-Host "S3FD checkpoint already exists: $FaceDetPath"
}

Write-Host "==> Patch audio.py for modern librosa"
$PatchTempFile = Join-Path $env:TEMP "patch_wav2lip_audio.py"
$PatchScript = @'
from pathlib import Path

audio_file = Path(r"__AUDIO_FILE__")
text = audio_file.read_text(encoding="utf-8")
old = "return librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels,"
new = "return librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft, n_mels=hp.num_mels,"
if old in text and new not in text:
    text = text.replace(old, new)
audio_file.write_text(text, encoding="utf-8")
print("audio.py patched")
'@.Replace("__AUDIO_FILE__", (Join-Path $RepoDir "audio.py").Replace("\", "\\"))
$PatchScript | Set-Content -LiteralPath $PatchTempFile -Encoding UTF8
Invoke-CondaRun -Args @("python", $PatchTempFile)
Remove-Item -LiteralPath $PatchTempFile -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Installation completed."
Write-Host "Repo:        $RepoDir"
Write-Host "Checkpoint:  $ModelPath"
Write-Host "ManualDrop:  $ManualModelPath"
Write-Host "FaceDet:     $FaceDetPath"
Write-Host ""
Write-Host "Start backend with:"
Write-Host "  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_web_app_torch.ps1"
