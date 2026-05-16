from pathlib import Path

import torch
from transformers import MarianConfig, MarianMTModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "Helsinki-NLP--opus-mt-zh-en"
BIN_PATH = MODEL_DIR / "pytorch_model.bin"
SAFE_PATH = MODEL_DIR / "model.safetensors"


def main() -> None:
    if SAFE_PATH.exists():
        print(f"ready: {SAFE_PATH}")
        return
    if not BIN_PATH.exists():
        raise FileNotFoundError(f"missing translation weights: {BIN_PATH}")

    config = MarianConfig.from_pretrained(MODEL_DIR)
    model = MarianMTModel(config)
    state_dict = torch.load(BIN_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    print(f"created: {SAFE_PATH}")


if __name__ == "__main__":
    main()
