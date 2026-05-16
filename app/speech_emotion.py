import os
import threading
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SPEECH_EMOTION_MODEL = os.getenv(
    "SPEECH_EMOTION_MODEL",
    "superb/wav2vec2-base-superb-er",
)
DEFAULT_SPEECH_EMOTION_CACHE_DIR = os.getenv(
    "SPEECH_EMOTION_CACHE_DIR",
    str(Path(__file__).resolve().parent.parent / "models" / "speech-emotion"),
)
SPEECH_EMOTION_MAX_SECONDS = float(os.getenv("SPEECH_EMOTION_MAX_SECONDS", "20"))

LABEL_MAP = {
    "ang": "愤怒",
    "anger": "愤怒",
    "angry": "愤怒",
    "hap": "高兴",
    "happy": "高兴",
    "joy": "高兴",
    "neu": "平静",
    "neutral": "平静",
    "sad": "悲伤",
    "sadness": "悲伤",
    "fear": "恐惧",
    "disgust": "厌恶",
    "surprise": "惊讶",
}


class SpeechEmotionRecognizer:
    def __init__(
        self,
        model_name_or_path: str = DEFAULT_SPEECH_EMOTION_MODEL,
        cache_dir: str = DEFAULT_SPEECH_EMOTION_CACHE_DIR,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()
        self._feature_extractor = None
        self._model = None

    def load(self) -> None:
        if self._model is not None and self._feature_extractor is not None:
            return
        with self._lock:
            if self._model is not None and self._feature_extractor is not None:
                return

            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            self._feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
            )
            self._model = AutoModelForAudioClassification.from_pretrained(
                self.model_name_or_path,
                cache_dir=self.cache_dir,
            )
            self._model.to(self.device)
            self._model.eval()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._feature_extractor is not None

    def predict_file(self, audio_path: str | Path, top_k: int = 5) -> dict[str, Any]:
        import librosa

        self.load()
        assert self._feature_extractor is not None
        assert self._model is not None

        target_sr = int(getattr(self._feature_extractor, "sampling_rate", 16000) or 16000)
        max_duration = max(1.0, SPEECH_EMOTION_MAX_SECONDS)
        waveform, sample_rate = librosa.load(
            str(audio_path),
            sr=target_sr,
            mono=True,
            duration=max_duration,
        )
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size < int(target_sr * 0.2):
            raise ValueError("音频太短，无法进行稳定的语音情感识别。")

        inputs = self._feature_extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = self._model(**inputs).logits[0]
            probabilities = torch.softmax(logits, dim=-1).detach().cpu().numpy()

        id2label = getattr(self._model.config, "id2label", {}) or {}
        ranked_indexes = np.argsort(probabilities)[::-1][:top_k]
        emotions = []
        for index in ranked_indexes:
            raw_label = str(id2label.get(int(index), f"LABEL_{int(index)}"))
            emotions.append(
                {
                    "name": normalize_speech_label(raw_label),
                    "label": raw_label,
                    "value": round(float(probabilities[index]) * 100.0, 1),
                }
            )

        return {
            "emotions": normalize_percentages(emotions),
            "model": self.model_name_or_path,
            "cache_dir": self.cache_dir,
            "device": self.device,
            "sample_rate": sample_rate,
            "duration_seconds": round(float(waveform.size) / float(sample_rate), 3),
        }


def normalize_speech_label(label: str) -> str:
    normalized = str(label or "").strip().lower()
    normalized = normalized.replace("label_", "")
    return LABEL_MAP.get(normalized, LABEL_MAP.get(normalized[:3], label or "未知"))


def normalize_percentages(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(max(0.0, float(item.get("value", 0.0))) for item in items)
    if total <= 0:
        return [{"name": "平静", "label": "neutral", "value": 100.0}]
    normalized = []
    for item in items:
        normalized.append(
            {
                **item,
                "value": round(max(0.0, float(item.get("value", 0.0))) / total * 100.0, 1),
            }
        )
    drift = round(100.0 - sum(float(item["value"]) for item in normalized), 1)
    if normalized and abs(drift) >= 0.1:
        normalized[0]["value"] = round(float(normalized[0]["value"]) + drift, 1)
    return normalized


speech_emotion_recognizer = SpeechEmotionRecognizer()
