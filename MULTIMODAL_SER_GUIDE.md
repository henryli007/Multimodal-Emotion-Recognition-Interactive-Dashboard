# 多模态语音情感识别扩展说明

## 功能目标

本项目保留原始文本链路：中文先经 Marian 翻译桥转为英文，再交给 GoEmotions 分类器；`Johnson8187/Chinese-Emotion-Small` 作为本地中文备用分类器。现在新增语音输入链路：用户在前端选择语音模式后，浏览器录音并尽量调用浏览器内置语音转写，后端使用语音情感识别模型分析音频情绪，再将文本情绪与语音情绪融合，形成多模态情绪画像。

文本输入时仍只使用文本情绪识别。语音输入时会同时展示语音情绪条形图，并在实时情绪感知面板中展示融合后的最终情绪。

## 模型选择

默认模型：

```text
Dpngtm/wav2vec2-emotion-recognition
```

选择理由：

- 直接提供 `model.safetensors`，能避开当前 `torch 2.3 + transformers 4.57` 对 `.bin` 权重加载的限制。
- 基于 Wav2Vec2，Transformers 原生支持 `AutoModelForAudioClassification`，部署路径简单。
- 模型规模适中，适合先在 4090D 环境中稳定落地。
- 标签覆盖常见语音情绪，便于前端做实时可视化展示。

相关模型页面：

- https://huggingface.co/superb/wav2vec2-base-superb-er
- https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition
- https://huggingface.co/speechbrain/emotion-recognition-wav2vec2-IEMOCAP

`superb/wav2vec2-base-superb-er` 只有 `pytorch_model.bin`，而当前 Transformers 会阻止 `torch<2.6` 读取这类权重，因此不再作为默认方案。SpeechBrain 的 IEMOCAP 模型在部分资料中标称 IEMOCAP 测试准确率约 78.7%，但集成方式和依赖更重；当前先选 Hugging Face Transformers 路线，部署风险更低。

## 环境变量

推荐将模型缓存放在数据盘，避免占用 root 主盘：

```bash
export SILICONFLOW_API_KEY=你的key
export SPEECH_EMOTION_MODEL=Dpngtm/wav2vec2-emotion-recognition
export SPEECH_EMOTION_CACHE_DIR=/root/autodl-tmp/models/speech-emotion
export OMP_NUM_THREADS=1
```

代理统一使用 7890：

```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5h://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export all_proxy=socks5h://127.0.0.1:7890
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

## 预下载语音情感模型

首次请求 `/speech_emotion` 会自动下载模型。也可以提前下载，避免演示时等待：

```bash
conda activate echomimic_v3
export SPEECH_EMOTION_CACHE_DIR=/root/autodl-tmp/models/speech-emotion

python - <<'PY'
from transformers import AutoProcessor, AutoModelForAudioClassification
model = "Dpngtm/wav2vec2-emotion-recognition"
cache_dir = "/root/autodl-tmp/models/speech-emotion"
AutoProcessor.from_pretrained(model, cache_dir=cache_dir)
AutoModelForAudioClassification.from_pretrained(
    model,
    cache_dir=cache_dir,
    use_safetensors=True,
)
print("speech emotion model cached:", cache_dir)
PY
```

## 启动服务

第一次启动前先把本地 `.bin` 翻译权重转换成当前环境可安全加载的 `safetensors`：

```bash
conda activate echomimic_v3
cd /root/autodl-tmp
python -m tools.prepare_translation_bridge
```

然后启动：

```bash
conda activate echomimic_v3
cd /root/autodl-tmp
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export SILICONFLOW_API_KEY=你的key
export TRANSLATION_MODEL_PATH=/root/autodl-tmp/models/Helsinki-NLP--opus-mt-zh-en
export ENGLISH_EMOTION_MODEL_PATH=/root/autodl-tmp/models/SamLowe--roberta-base-go_emotions
export TEXT_EMOTION_MODEL=Johnson8187/Chinese-Emotion-Small
export TEXT_EMOTION_CACHE_DIR=/root/autodl-tmp/models/text-emotion
export SPEECH_EMOTION_MODEL=/root/autodl-tmp/models/speech-emotion-direct
export OMP_NUM_THREADS=1
uvicorn app.web_app:app --host 0.0.0.0 --port 8000
```

## 在本地电脑查看前端

服务器没有浏览器界面时，用 SSH 本地端口转发。假设 AutoDL SSH 端口是 `服务器端口`：

```bash
ssh -N -L 8000:127.0.0.1:8000 root@服务器地址 -p 服务器端口
```

然后在本地电脑浏览器打开：

```text
http://127.0.0.1:8000
```

如果前端要用麦克风，建议使用 `localhost/127.0.0.1` 访问，因为浏览器通常只允许 HTTPS 或 localhost 页面调用麦克风权限。

## 接口

语音情感识别：

```bash
curl -X POST http://127.0.0.1:8000/speech_emotion \
  -F "audio=@sample.wav"
```

聊天接口支持多模态输入：

```json
{
  "message": "我最近压力很大",
  "input_mode": "voice",
  "speech_emotions": [
    {"name": "紧张", "value": 62.1},
    {"name": "平静", "value": 21.4}
  ],
  "speech_emotion_model": "Dpngtm/wav2vec2-emotion-recognition"
}
```

返回中会包含：

- `emotions`: 融合后的最终情绪分布
- `text_emotions`: 文本情绪分布
- `speech_emotions`: 语音情绪分布
- `emotion_mode`: `text` 或 `multimodal`

## 汇报图生成

生成顶会论文风格的中文 SVG 图：

```bash
python -m tools.generate_report_figures
```

输出目录：

```text
docs/figures/
```

包含：

- `figure_1_case_trace.svg`: 单句输入在系统中的运行产物
- `figure_2_multimodal_fusion.svg`: 文本、语音与融合情绪对照
- `figure_3_graph_evidence.svg`: 真实图谱证据子图

SVG 可以直接插入 PPT、论文或用浏览器打开。若需要 PNG，可在本地用浏览器、Inkscape 或设计软件导出，能保持中文字体质量。

## 推送准备

根目录准备作为新的 Git 仓库推送到：

```text
git@github.com:henryli007/Multimodal-Emotion-Recognition-Interactive-Dashboard.git
```

注意：

- 模型权重、生成媒体、上传音频、缓存和密钥文件已通过 `.gitignore` 排除。
- `SILICONFLOW_API_KEY` 已改为环境变量读取，不应提交真实 key。
- 当前服务器缺少 `gh`，且 SSH 访问 GitHub 时返回 `Permission denied (publickey)`。需要先配置 SSH key 或 token 后才能实际 push。
