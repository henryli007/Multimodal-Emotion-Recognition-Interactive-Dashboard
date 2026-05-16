#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="$ROOT_DIR/EchoMimicV3"
ENV_NAME="${ENV_NAME:-echomimic_v3}"
FLASH_ROOT="$REPO_DIR/flash-pro"
WAN_DIR="$FLASH_ROOT/Wan2.1-Fun-V1.1-1.3B-InP"
WAV2VEC_DIR="$FLASH_ROOT/chinese-wav2vec2-base"
TRANSFORMER_DIR="$FLASH_ROOT/transformer"

if [ -f /etc/network_turbo ]; then
  # This host uses a helper script for faster access to GitHub and Hugging Face.
  source /etc/network_turbo || true
fi

if [ ! -d "$REPO_DIR" ]; then
  echo "EchoMimicV3 repo not found at $REPO_DIR" >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  CONDA_NO_PLUGINS=true conda create -y -n "$ENV_NAME" python=3.10 --override-channels -c defaults --solver classic
fi

conda activate "$ENV_NAME"

PIP_FLAGS=(--retries 10 --timeout 120)

python -m pip install "${PIP_FLAGS[@]}" --upgrade pip setuptools wheel
python -m pip install "${PIP_FLAGS[@]}" \
  filelock \
  fsspec \
  jinja2 \
  markupsafe \
  mpmath==1.3.0 \
  networkx \
  numpy \
  pillow \
  sympy==1.13.1 \
  typing-extensions
python -m pip install "${PIP_FLAGS[@]}" torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -m pip install "${PIP_FLAGS[@]}" -r "$REPO_DIR/requirements.txt"
python -m pip install "${PIP_FLAGS[@]}" huggingface_hub modelscope pyloudnorm

mkdir -p "$FLASH_ROOT" "$TRANSFORMER_DIR"

export WAN_DIR WAV2VEC_DIR TRANSFORMER_DIR
python - <<'PY'
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download
from modelscope import snapshot_download as ms_snapshot_download

wan_dir = Path(os.environ["WAN_DIR"])
wav2vec_dir = Path(os.environ["WAV2VEC_DIR"])
transformer_dir = Path(os.environ["TRANSFORMER_DIR"])

wan_dir.parent.mkdir(parents=True, exist_ok=True)
wav2vec_dir.parent.mkdir(parents=True, exist_ok=True)
transformer_dir.mkdir(parents=True, exist_ok=True)

required_wan_files = [
    wan_dir / "config.json",
    wan_dir / "diffusion_pytorch_model.safetensors",
    wan_dir / "models_t5_umt5-xxl-enc-bf16.pth",
    wan_dir / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    wan_dir / "Wan2.1_VAE.pth",
]
if not all(path.exists() for path in required_wan_files):
    snapshot_download(
        repo_id="alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
        local_dir=str(wan_dir),
        local_dir_use_symlinks=False,
    )

transformer_file = transformer_dir / "diffusion_pytorch_model.safetensors"
min_transformer_size = 3_000_000_000
if not transformer_file.exists() or transformer_file.stat().st_size < min_transformer_size:
    cache_dir = Path(
        snapshot_download(
            repo_id="BadToBest/EchoMimicV3",
            allow_patterns=[
                "transformer/diffusion_pytorch_model.safetensors",
                "echomimicv3-flash-pro/diffusion_pytorch_model.safetensors",
                "echomimicv3-flash-pro/transformer/diffusion_pytorch_model.safetensors",
            ],
        )
    )
    source_candidates = [
        cache_dir / "transformer" / "diffusion_pytorch_model.safetensors",
        cache_dir / "echomimicv3-flash-pro" / "diffusion_pytorch_model.safetensors",
        cache_dir / "echomimicv3-flash-pro" / "transformer" / "diffusion_pytorch_model.safetensors",
    ]
    source_file = next((path for path in source_candidates if path.exists()), None)
    if source_file is None:
        raise FileNotFoundError(f"Missing transformer weight in snapshot: {cache_dir}")
    shutil.copy2(source_file, transformer_file)

if not (wav2vec_dir / "config.json").exists():
    ms_snapshot_download("TencentGameMate/chinese-wav2vec2-base", local_dir=str(wav2vec_dir))
PY

echo
echo "EchoMimicV3 Flash setup finished."
echo "Detected web app defaults:"
echo "  Repo: $REPO_DIR"
echo "  Env : $ENV_NAME"
echo "  Weights: $FLASH_ROOT"
echo
echo "Recommended run command:"
echo "  source \$(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME && python $ROOT_DIR/web_app.py"
