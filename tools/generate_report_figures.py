from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "docs" / "figures"


STYLE = """
<style>
svg { background: #fbfcfb; }
.title { font: 700 26px "Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif; fill: #18352b; }
.subtitle { font: 500 14px "Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif; fill: #60766a; }
.label { font: 700 15px "Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif; fill: #29483a; }
.small { font: 500 12px "Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif; fill: #61766a; }
.tiny { font: 500 10px "Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif; fill: #718379; }
.box { fill: #ffffff; stroke: #d9e5dc; stroke-width: 1.2; rx: 12; }
.soft { fill: #eef7f1; stroke: #bad9c6; stroke-width: 1.1; rx: 12; }
.audio { fill: #fff1e7; stroke: #efbd9d; stroke-width: 1.1; rx: 12; }
.text { fill: #edf5ff; stroke: #b9d1ef; stroke-width: 1.1; rx: 12; }
.fusion { fill: #eff4f0; stroke: #7db895; stroke-width: 1.6; rx: 16; }
.line { stroke: #8aa99a; stroke-width: 1.8; fill: none; marker-end: url(#arrow); }
.dash { stroke: #c7d6cd; stroke-width: 1.2; stroke-dasharray: 5 5; fill: none; }
.barbg { fill: #e7eee9; rx: 4; }
.bar1 { fill: #76b88f; rx: 4; }
.bar2 { fill: #efaa80; rx: 4; }
.bar3 { fill: #8bb9e6; rx: 4; }
</style>
"""


def wrap(text: str, chars: int = 18) -> list[str]:
    return [text[index : index + chars] for index in range(0, len(text), chars)] or [""]


def text_block(x: int, y: int, lines: list[str], klass: str = "small", line_height: int = 18) -> str:
    return "\n".join(
        f'<text class="{klass}" x="{x}" y="{y + i * line_height}">{line}</text>'
        for i, line in enumerate(lines)
    )


def defs() -> str:
    return """
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#8aa99a"/>
  </marker>
  <filter id="shadow" x="-20%" y="-20%" width="140%" height="150%">
    <feDropShadow dx="0" dy="7" stdDeviation="8" flood-color="#8ba696" flood-opacity="0.18"/>
  </filter>
</defs>
"""


def architecture_svg() -> str:
    boxes = [
        (70, 120, 170, 88, "text", "文本输入", ["中文咨询内容", "RoBERTa GoEmotions"]),
        (70, 250, 170, 88, "audio", "语音输入", ["浏览器录音/转写", "Wav2Vec2 SER"]),
        (315, 120, 190, 218, "fusion", "多模态情绪层", ["文本情绪分布", "语音情绪分布", "加权融合与归一化", "输出统一情绪画像"]),
        (585, 112, 190, 96, "soft", "LightRAG 知识图谱", ["1729 节点", "2010 关系", "心理干预知识"]),
        (585, 248, 190, 96, "soft", "咨询生成与播报", ["SiliconFlow LLM", "edge-tts", "EchoMimicV3/Wav2Lip"]),
        (835, 176, 210, 118, "box", "交互式前端", ["情绪环图", "语音情绪条形图", "知识子图", "数字人反馈"]),
    ]
    body = []
    for x, y, w, h, klass, title, lines in boxes:
        body.append(f'<rect class="{klass}" filter="url(#shadow)" x="{x}" y="{y}" width="{w}" height="{h}"/>')
        body.append(f'<text class="label" x="{x + 18}" y="{y + 30}">{title}</text>')
        body.append(text_block(x + 18, y + 56, lines, "small", 18))
    lines = [
        (240, 164, 315, 174),
        (240, 294, 315, 282),
        (505, 176, 585, 160),
        (505, 282, 585, 296),
        (775, 160, 835, 210),
        (775, 296, 835, 250),
    ]
    for x1, y1, x2, y2 in lines:
        body.append(f'<path class="line" d="M{x1},{y1} C{x1 + 35},{y1} {x2 - 35},{y2} {x2},{y2}"/>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="440" viewBox="0 0 1120 440">
{defs()}{STYLE}
<text class="title" x="70" y="58">多模态心理陪伴系统架构</text>
<text class="subtitle" x="70" y="84">文本、语音、知识图谱与数字人播报在同一服务链路内闭环</text>
{''.join(body)}
</svg>"""


def fusion_svg() -> str:
    labels = ["悲伤", "紧张", "平静", "愤怒", "宽慰"]
    text_values = [36, 24, 18, 12, 10]
    speech_values = [18, 39, 22, 14, 7]
    fused_values = [28, 31, 20, 13, 8]
    body = []
    y0 = 120
    for i, label in enumerate(labels):
        y = y0 + i * 58
        body.append(f'<text class="label" x="88" y="{y + 14}">{label}</text>')
        for x, value, klass in [(190, text_values[i], "bar3"), (430, speech_values[i], "bar2"), (690, fused_values[i], "bar1")]:
            body.append(f'<rect class="barbg" x="{x}" y="{y}" width="180" height="16"/>')
            body.append(f'<rect class="{klass}" x="{x}" y="{y}" width="{value * 1.8}" height="16"/>')
            body.append(f'<text class="small" x="{x + 192}" y="{y + 13}">{value}%</text>')
    headers = [(190, "文本情绪"), (430, "语音情绪"), (690, "融合情绪")]
    for x, label in headers:
        body.append(f'<text class="label" x="{x}" y="94">{label}</text>')
    body.append('<path class="dash" d="M630,85 L630,395"/>')
    body.append('<rect class="fusion" x="910" y="140" width="145" height="145" filter="url(#shadow)"/>')
    body.append('<text class="label" x="940" y="185">融合策略</text>')
    body.append(text_block(932, 216, ["文本权重 0.55", "语音权重 0.45", "同名情绪累加", "归一化到 100%"], "small", 22))
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="440" viewBox="0 0 1120 440">
{defs()}{STYLE}
<text class="title" x="70" y="58">文本-语音情绪融合示意</text>
<text class="subtitle" x="70" y="84">语音输入时同时展示单模态分布和融合后的最终情绪画像</text>
{''.join(body)}
</svg>"""


def benchmark_svg() -> str:
    metrics = [
        ("GPU", "RTX 4090D / 24GB", 96, "bar1"),
        ("文本情绪", "RoBERTa + 翻译模型", 82, "bar3"),
        ("语音情绪", "Wav2Vec2 SUPERB-ER", 78, "bar2"),
        ("知识图谱", "1729 节点 / 2010 关系", 88, "bar1"),
        ("数字人", "EchoMimicV3 优先回退 Wav2Lip", 74, "bar2"),
    ]
    body = []
    for i, (name, desc, value, klass) in enumerate(metrics):
        y = 128 + i * 58
        body.append(f'<text class="label" x="86" y="{y + 13}">{name}</text>')
        body.append(f'<text class="small" x="195" y="{y + 13}">{desc}</text>')
        body.append(f'<rect class="barbg" x="500" y="{y}" width="360" height="18"/>')
        body.append(f'<rect class="{klass}" x="500" y="{y}" width="{value * 3.6}" height="18"/>')
        body.append(f'<text class="label" x="885" y="{y + 15}">{value}</text>')
    body.append('<rect class="soft" x="70" y="368" width="980" height="42"/>')
    body.append('<text class="small" x="92" y="394">注：该图用于汇报展示，数值为系统能力画像与资源适配指标；正式论文实验可替换为真实评测结果。</text>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="450" viewBox="0 0 1120 450">
{defs()}{STYLE}
<text class="title" x="70" y="58">系统能力与资源画像</text>
<text class="subtitle" x="70" y="84">覆盖硬件适配、情绪识别、图谱检索与数字人生成能力</text>
{''.join(body)}
</svg>"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures = {
        "figure_1_architecture.svg": architecture_svg(),
        "figure_2_emotion_fusion.svg": fusion_svg(),
        "figure_3_system_profile.svg": benchmark_svg(),
    }
    for name, content in figures.items():
        (OUTPUT_DIR / name).write_text(content, encoding="utf-8")
        print(OUTPUT_DIR / name)


if __name__ == "__main__":
    main()
