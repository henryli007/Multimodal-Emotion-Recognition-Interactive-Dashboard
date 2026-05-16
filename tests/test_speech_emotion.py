from app.speech_emotion import normalize_percentages, normalize_speech_label
from app.web_app import merge_emotion_sources


def test_normalize_speech_label_maps_common_ser_labels():
    assert normalize_speech_label("ang") == "愤怒"
    assert normalize_speech_label("hap") == "高兴"
    assert normalize_speech_label("neu") == "平静"
    assert normalize_speech_label("sad") == "悲伤"


def test_normalize_percentages_sums_to_100():
    result = normalize_percentages(
        [
            {"name": "愤怒", "label": "ang", "value": 2},
            {"name": "平静", "label": "neu", "value": 1},
        ]
    )

    assert result[0]["value"] == 66.7
    assert result[1]["value"] == 33.3
    assert round(sum(item["value"] for item in result), 1) == 100.0


def test_merge_emotion_sources_combines_text_and_speech():
    merged = merge_emotion_sources(
        [{"name": "悲伤", "value": 70}, {"name": "平静", "value": 30}],
        [{"name": "紧张", "value": 80}, {"name": "平静", "value": 20}],
    )

    assert merged[0]["name"] in {"悲伤", "紧张"}
    assert any(item["name"] == "平静" for item in merged)
    assert round(sum(item["value"] for item in merged), 1) == 100.0
