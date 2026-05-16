# AI 心理陪伴多模态交互面板

这是一个基于 `FastAPI` 的心理陪伴多模态演示项目，整合了：

- 情绪识别
- `LightRAG` 心理知识图谱检索
- SiliconFlow 大模型问答
- `edge-tts` 中文语音合成
- 数字人口型视频生成
- 图谱可视化
- 基于 `PsyQA` 的 LoRA 微调脚本

这次已经按职责重新整理了目录，核心应用代码放在 `app/`，辅助脚本放在 `tools/`，数据集中放在 `data/`，图谱工作区放在 `workspace/`。根目录仍然保留了兼容入口，所以原来的常见命令还能继续使用。

## 当前目录结构

```text
.
├── app/                     # 主应用代码
│   ├── web_app.py
│   └── knowledge_base.py
├── tools/                   # 辅助脚本
│   ├── build_kg.py
│   ├── finetune.py
│   └── visualize_graph.py
├── data/                    # 项目数据
│   ├── knowledge/
│   │   └── data_pro.json
│   └── psyqa/
│       ├── psyqa_train.jsonl
│       ├── psyqa_validation.jsonl
│       └── psyqa_test.jsonl
├── workspace/               # 运行期工作区
│   ├── lightrag/
│   └── knowledge_graph_visualization.html
├── static/                  # 前端页面、头像、生成媒体
├── models/                  # 本地模型目录
├── scripts/                 # 环境搭建 / PowerShell 辅助脚本
├── EchoMimicV3/             # 可选数字人后端 1
├── Wav2Lip/                 # 可选数字人后端 2
├── web_app.py               # 兼容启动入口
├── build_kg.py              # 兼容图谱构建入口
├── finetune.py              # 兼容训练入口
├── visualize_graph.py       # 兼容可视化入口
├── requirements.txt
└── README.md
```

## 主要模块

### `app/`

- [app/web_app.py](/root/autodl-tmp/app/web_app.py)
  - 主 Web 服务
  - 提供 `/`、`/graph_data`、`/chat` 接口
  - 串联情绪识别、RAG 检索、LLM 回复、TTS、数字人生成
- [app/knowledge_base.py](/root/autodl-tmp/app/knowledge_base.py)
  - 初始化 `LightRAG`
  - 配置 SiliconFlow 的聊天与向量接口
  - 工作区路径已统一到 `workspace/lightrag/`

### `tools/`

- [tools/build_kg.py](/root/autodl-tmp/tools/build_kg.py)
  - 从 `data/knowledge/data_pro.json` 构建或增量更新图谱
- [tools/finetune.py](/root/autodl-tmp/tools/finetune.py)
  - 使用 `data/psyqa/psyqa_train.jsonl` 做 LoRA 微调
- [tools/visualize_graph.py](/root/autodl-tmp/tools/visualize_graph.py)
  - 读取 `workspace/lightrag/graph_chunk_entity_relation.graphml`
  - 生成 `workspace/knowledge_graph_visualization.html`

### `data/`

- `data/knowledge/`
  - 心理知识图谱原始数据
- `data/psyqa/`
  - 微调数据集

### `workspace/`

- `workspace/lightrag/`
  - `LightRAG` 的图谱、向量索引、缓存文件
- `workspace/knowledge_graph_visualization.html`
  - 图谱可视化产物

## 环境要求

- Python 3.10 及以上
- 推荐 Linux + NVIDIA GPU
- 已安装 `ffmpeg`
- 若启用数字人视频，需准备对应模型与权重

## 安装依赖

安装主项目依赖：

```bash
pip install -r requirements.txt
```

如果你要启用数字人后端，再按需安装：

```bash
pip install -r EchoMimicV3/requirements.txt
pip install -r Wav2Lip/requirements.txt
```

## 模型与资源准备

代码默认会读取以下本地资源：

- `models/Helsinki-NLP--opus-mt-zh-en`
  - 中文转英文翻译模型
- `models/SamLowe--roberta-base-go_emotions`
  - 情绪分类模型
- `models/Qwen2.5-1.5B`
  - LoRA 微调基础模型
- `static/image.png`
  - 默认数字人驱动头像

如果启用 `Wav2Lip`，还需要：

- `Wav2Lip/checkpoints/wav2lip.pth` 或 `wav2lip_gan.pth`
- `Wav2Lip/face_detection/detection/sfd/s3fd.pth`

如果启用 `EchoMimicV3`，需要准备其 `flash-pro` 权重目录和 `wav2vec` 模型目录。

## 环境变量

常用环境变量：

| 变量名 | 说明 | 是否必需 |
| --- | --- | --- |
| `SILICONFLOW_API_KEY` | SiliconFlow API Key | 是 |
| `SOURCE_IMAGE` | 数字人驱动头像路径 | 否 |
| `WAV2LIP_DIR` | `Wav2Lip` 项目目录 | 否 |
| `WAV2LIP_PYTHON` | `Wav2Lip` Python 路径 | 否 |
| `WAV2LIP_CHECKPOINT` | `Wav2Lip` 权重路径 | 否 |
| `WAV2LIP_FACE_DET` | `s3fd.pth` 路径 | 否 |
| `ECHOMIMIC_V3_DIR` | `EchoMimicV3` 项目目录 | 否 |
| `ECHOMIMIC_V3_PYTHON` | `EchoMimicV3` Python 路径 | 否 |
| `ECHOMIMIC_V3_CONFIG` | EchoMimicV3 配置路径 | 否 |
| `ECHOMIMIC_V3_MODEL_ROOT` | EchoMimicV3 权重根目录 | 否 |

## 推荐运行方式

### 1. 构建知识图谱

推荐：

```bash
python -m tools.build_kg --data ./data/knowledge/data_pro.json
```

兼容旧入口：

```bash
python build_kg.py --data ./data/knowledge/data_pro.json
```

构建结果会写入：

```text
./workspace/lightrag/
```

### 2. 启动服务

推荐：

```bash
uvicorn app.web_app:app --host 0.0.0.0 --port 8000
```

兼容旧入口：

```bash
python web_app.py
```

启动后访问：

```text
http://127.0.0.1:8000
```

### 3. 图谱可视化

推荐：

```bash
python -m tools.visualize_graph
```

兼容旧入口：

```bash
python visualize_graph.py
```

输出文件：

```text
./workspace/knowledge_graph_visualization.html
```

### 4. LoRA 微调

推荐：

```bash
python -m tools.finetune
```

兼容旧入口：

```bash
python finetune.py
```

默认使用：

- 基础模型：`./models/Qwen2.5-1.5B`
- 数据集：`./data/psyqa/psyqa_train.jsonl`
- 输出目录：`./models/Qwen2.5-1.5B-PsyQA-LoRA`

## 接口示例

### 获取图谱数据

```bash
curl http://127.0.0.1:8000/graph_data
```

### 发起聊天请求

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"最近考试压力很大，总觉得睡不好。"}'
```

返回内容主要包括：

- `reply`
- `speech_text`
- `media_url`
- `media_type`
- `emotions`
- `text_emotions`
- `speech_emotions`
- `emotion_mode`
- `highlight_nodes`
- `highlight_edges`

### 语音情感识别

语音输入时，前端会将录音上传到 `/speech_emotion`，后端使用默认模型
`superb/wav2vec2-base-superb-er` 识别语音情绪。模型缓存建议放在数据盘：

```bash
export SPEECH_EMOTION_MODEL=superb/wav2vec2-base-superb-er
export SPEECH_EMOTION_CACHE_DIR=/root/autodl-tmp/models/speech-emotion
```

也可以直接测试接口：

```bash
curl -X POST http://127.0.0.1:8000/speech_emotion \
  -F "audio=@sample.wav"
```

语音模式下 `/chat` 会融合文本情绪和语音情绪；文本模式仍只使用文本情绪识别。

更多说明见 [MULTIMODAL_SER_GUIDE.md](/root/autodl-tmp/MULTIMODAL_SER_GUIDE.md)。

### 汇报图生成

项目提供中文 SVG 汇报图生成脚本，用于展示系统架构、多模态情绪融合和资源画像：

```bash
python -m tools.generate_report_figures
```

输出目录：

```text
docs/figures/
```

## 数字人后端回退顺序

主服务会按以下顺序尝试视频生成：

1. `EchoMimicV3`
2. `Wav2Lip`
3. `Ditto`
4. `LivePortrait`

如果都不可用，会自动退回语音模式。

## 运行注意事项

- `ffmpeg` 未安装时，`Wav2Lip` 音频预处理会失败
- 若未找到数字人后端，系统会自动回退到语音模式
- 生成的语音和视频会写入 `static/media/`
- `LightRAG` 工作区现在统一位于 `workspace/lightrag/`

## 后续建议

- 把敏感配置统一迁移到环境变量或 `.env`
- 增加 `conda` 环境文件或锁定版依赖文件
- 为 `EchoMimicV3`、`Wav2Lip`、`Ditto`、`LivePortrait` 单独补接入文档
- 为接口增加错误响应示例和部署说明
