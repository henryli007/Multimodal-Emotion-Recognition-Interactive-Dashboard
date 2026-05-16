import asyncio
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import aiohttp
import edge_tts
import networkx as nx
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response
from transformers import AutoTokenizer, pipeline

from app.knowledge_base import SILICONFLOW_API_KEY as KB_SILICONFLOW_API_KEY
from app.knowledge_base import rag as kb_rag
from app.speech_emotion import speech_emotion_recognizer
from lightrag import QueryParam


PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKING_DIR = str(PROJECT_ROOT / "workspace" / "lightrag")
STATIC_DIR = str(PROJECT_ROOT / "static")
MEDIA_DIR = str(Path(STATIC_DIR) / "media")
SOURCE_IMAGE = os.getenv("SOURCE_IMAGE", str(Path(STATIC_DIR) / "image.png"))
GRAPHML_PATH = str(Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml")

SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY") or KB_SILICONFLOW_API_KEY
WAV2LIP_DIR = os.getenv("WAV2LIP_DIR", "")
WAV2LIP_PYTHON = os.getenv("WAV2LIP_PYTHON", "")
WAV2LIP_CHECKPOINT = os.getenv("WAV2LIP_CHECKPOINT", "")
WAV2LIP_FACE_DET = os.getenv("WAV2LIP_FACE_DET", "")
WAV2LIP_FACE_DET_BATCH_SIZE = int(os.getenv("WAV2LIP_FACE_DET_BATCH_SIZE", "2"))
WAV2LIP_BATCH_SIZE = int(os.getenv("WAV2LIP_BATCH_SIZE", "4"))
WAV2LIP_RESIZE_FACTOR = int(os.getenv("WAV2LIP_RESIZE_FACTOR", "1"))
WAV2LIP_FPS = float(os.getenv("WAV2LIP_FPS", "25"))
WAV2LIP_PADS = os.getenv("WAV2LIP_PADS", "0 20 0 0")
DITTO_DIR = os.getenv("DITTO_DIR", "")
DITTO_SCRIPT = os.getenv("DITTO_SCRIPT", "")
DITTO_CHECKPOINT_ROOT = os.getenv("DITTO_CHECKPOINT_ROOT", "")
DITTO_CFG_PKL = os.getenv("DITTO_CFG_PKL", "")
DITTO_MODEL_ROOT = os.getenv("DITTO_MODEL_ROOT", "")
DITTO_PYTHON = os.getenv("DITTO_PYTHON", "")
LIVEPORTRAIT_DIR = os.getenv("LIVEPORTRAIT_DIR", "")
LIVEPORTRAIT_SCRIPT = os.getenv("LIVEPORTRAIT_SCRIPT", "")
LIVEPORTRAIT_EXTRA_ARGS = os.getenv("LIVEPORTRAIT_EXTRA_ARGS", "")
ECHOMIMIC_V3_DIR = os.getenv("ECHOMIMIC_V3_DIR", "")
ECHOMIMIC_V3_SCRIPT = os.getenv("ECHOMIMIC_V3_SCRIPT", "")
ECHOMIMIC_V3_PYTHON = os.getenv("ECHOMIMIC_V3_PYTHON", "")
ECHOMIMIC_V3_CONFIG = os.getenv("ECHOMIMIC_V3_CONFIG", "")
ECHOMIMIC_V3_MODEL_ROOT = os.getenv("ECHOMIMIC_V3_MODEL_ROOT", "")
ECHOMIMIC_V3_TRANSFORMER_PATH = os.getenv("ECHOMIMIC_V3_TRANSFORMER_PATH", "")
ECHOMIMIC_V3_WAV2VEC_DIR = os.getenv("ECHOMIMIC_V3_WAV2VEC_DIR", "")
ECHOMIMIC_V3_PROMPT = os.getenv(
    "ECHOMIMIC_V3_PROMPT",
    "A person is speaking calmly with natural lip sync, subtle head motion, stable framing, and preserved background.",
)
ECHOMIMIC_V3_GUIDANCE_SCALE = float(os.getenv("ECHOMIMIC_V3_GUIDANCE_SCALE", "5.5"))
ECHOMIMIC_V3_AUDIO_GUIDANCE_SCALE = float(os.getenv("ECHOMIMIC_V3_AUDIO_GUIDANCE_SCALE", "3.0"))
ECHOMIMIC_V3_AUDIO_SCALE = float(os.getenv("ECHOMIMIC_V3_AUDIO_SCALE", "1.0"))
ECHOMIMIC_V3_SHIFT = float(os.getenv("ECHOMIMIC_V3_SHIFT", "5.0"))
ECHOMIMIC_V3_STEPS = int(os.getenv("ECHOMIMIC_V3_STEPS", "15"))
ECHOMIMIC_V3_FPS = int(os.getenv("ECHOMIMIC_V3_FPS", "25"))
ECHOMIMIC_V3_MAX_FRAMES = int(os.getenv("ECHOMIMIC_V3_MAX_FRAMES", "0"))
ECHOMIMIC_V3_PARTIAL_VIDEO_LENGTH = int(os.getenv("ECHOMIMIC_V3_PARTIAL_VIDEO_LENGTH", "49"))
ECHOMIMIC_V3_OVERLAP_VIDEO_LENGTH = int(os.getenv("ECHOMIMIC_V3_OVERLAP_VIDEO_LENGTH", "4"))
ECHOMIMIC_V3_SAMPLE_SIZE = os.getenv("ECHOMIMIC_V3_SAMPLE_SIZE", "512 512")
ECHOMIMIC_V3_WEIGHT_DTYPE = os.getenv("ECHOMIMIC_V3_WEIGHT_DTYPE", "bfloat16")
ECHOMIMIC_V3_GPU_MEMORY_MODE = os.getenv("ECHOMIMIC_V3_GPU_MEMORY_MODE", "sequential_cpu_offload")
ECHOMIMIC_V3_EXTRA_ARGS = os.getenv("ECHOMIMIC_V3_EXTRA_ARGS", "")
os.makedirs(MEDIA_DIR, exist_ok=True)
UPLOAD_DIR = str(PROJECT_ROOT / "workspace" / "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

GRAPH_TERM_ALIASES: dict[str, list[str]] = {
    "pressure": ["压力"],
    "stress": ["压力", "应激"],
    "exam": ["考试"],
    "exams": ["考试"],
    "study": ["学习"],
    "work": ["工作"],
    "anxiety": ["焦虑"],
    "sleep": ["睡眠"],
    "insomnia": ["失眠"],
    "fatigue": ["疲惫"],
    "emotion": ["情绪"],
    "mood": ["情绪"],
    "depression": ["抑郁"],
    "social": ["社交"],
    "support": ["支持"],
    "selfcare": ["自我关怀"],
    "self-care": ["自我关怀"],
    "breathing": ["呼吸", "深呼吸"],
    "relaxation": ["放松"],
    "stressmanagement": ["压力管理", "情绪调节"],
}

_graph_snapshot_cache: dict[str, Any] = {"mtime": None, "snapshot": None}
ECHOMIMIC_PROGRESS_RE = re.compile(r"^EMV3_PROGRESS\|phase=(.*?)\|percent=([0-9]+(?:\.[0-9]+)?)\|detail=(.*)$")
logger = logging.getLogger(__name__)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", str(value).strip().lower())


def format_duration_compact(seconds: float | int) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def render_terminal_progress(prefix: str, percent: float, detail: str, start_time: float) -> None:
    percent = max(0.0, min(100.0, float(percent)))
    elapsed = max(0.0, time.time() - start_time)
    eta_text = "--:--"
    if percent > 0.0:
        eta_seconds = elapsed * (100.0 - percent) / percent
        eta_text = format_duration_compact(eta_seconds)

    terminal_width = shutil.get_terminal_size((120, 20)).columns
    bar_width = max(18, min(34, terminal_width - len(prefix) - 42))
    filled = min(bar_width, max(0, int(round(bar_width * percent / 100.0))))
    bar = "#" * filled + "-" * (bar_width - filled)
    detail_text = str(detail or "").strip()
    max_detail_len = max(24, terminal_width - len(prefix) - bar_width - 28)
    if len(detail_text) > max_detail_len:
        detail_text = detail_text[: max_detail_len - 1] + "…"

    sys.stdout.write(
        f"\r{prefix} [{bar}] {percent:5.1f}% ETA {eta_text} | {detail_text:<{max_detail_len}}"
    )
    sys.stdout.flush()


def run_process_with_terminal_progress(
    command: list[str],
    cwd: Path | str,
    env: dict[str, str],
    prefix: str,
) -> tuple[int, str]:
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    start_time = time.time()
    captured_lines: list[str] = []
    progress_rendered = False

    assert process.stdout is not None
    for raw_line in process.stdout:
        captured_lines.append(raw_line)
        stripped = raw_line.rstrip("\n")
        progress_match = ECHOMIMIC_PROGRESS_RE.match(stripped.strip())
        if progress_match:
            _, percent_text, detail_text = progress_match.groups()
            render_terminal_progress(prefix, float(percent_text), detail_text, start_time)
            progress_rendered = True
            continue

        if progress_rendered:
            sys.stdout.write("\n")
            sys.stdout.flush()
            progress_rendered = False

        if stripped.strip():
            print(f"{prefix} {stripped}")

    return_code = process.wait()
    if progress_rendered:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return return_code, "".join(captured_lines)


def rank_text_items(items: list[str], limit: int) -> list[str]:
    counter: Counter[str] = Counter()
    display_map: dict[str, str] = {}
    for item in items:
        cleaned = str(item).strip().strip("[](){}'\"")
        norm = normalize_text(cleaned)
        if not norm:
            continue
        counter[norm] += 1
        display_map.setdefault(norm, cleaned)
    return [display_map[key] for key, _ in counter.most_common(limit)]


def stringify_knowledge(payload: Any) -> str:
    if payload is None:
        return "暂无相关图谱资料。"

    if isinstance(payload, str):
        return payload

    if isinstance(payload, dict):
        for key in ("response", "content", "text", "answer", "description"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(payload, ensure_ascii=False)

    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
            elif isinstance(item, dict):
                extracted = None
                for key in ("response", "content", "text", "answer", "description"):
                    value = item.get(key)
                    if isinstance(value, str) and value.strip():
                        extracted = value.strip()
                        break
                parts.append(extracted or json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part).strip() or "暂无相关图谱资料。"

    return str(payload)


def split_logged_terms(raw_groups: list[str]) -> list[str]:
    terms: list[str] = []
    for group in raw_groups:
        if not group:
            continue
        parts = re.split(r"[;,；]", str(group))
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                terms.append(cleaned)
    return terms


def normalize_generated_reply(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return "抱歉，我暂时没能整理出合适的回应。"

    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^[ \t]*[-*+][ \t]+", "", cleaned, flags=re.M)
    cleaned = re.sub(r"^[ \t]*\d+[\.、．)\]][ \t]*", "", cleaned, flags=re.M)
    cleaned = re.sub(r"[ \t\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.replace(", ", "，").replace(",", "，")
    cleaned = cleaned.replace("; ", "；").replace(";", "；")
    cleaned = cleaned.replace(": ", "：").replace(":", "：")
    cleaned = cleaned.replace("! ", "！").replace("!", "！")
    cleaned = cleaned.replace("? ", "？").replace("?", "？")
    cleaned = cleaned.replace(". ", "。").replace("..", "。")
    cleaned = re.sub(r"([，。！？；：、])\1+", r"\1", cleaned)
    cleaned = re.sub(r"([，；：、])([。！？])", r"\2", cleaned)
    cleaned = re.sub(r"([。！？])([，；：、])", r"\1", cleaned)
    cleaned = re.sub(r"\s*([，。！？；：、])\s*", r"\1", cleaned)

    # Prefer readable Chinese range wording such as "5到10分钟".
    cleaned = re.sub(r"(\d{1,2})\s*-\s*(\d{1,2})(分钟|小时|天|周|次)", r"\1到\2\3", cleaned)

    # Fix common malformed duplicated-leading-1 ranges such as "5-110分钟" -> "5到10分钟".
    def _fix_malformed_range(match: re.Match[str]) -> str:
        left = match.group(1)
        right = match.group(2)
        unit = match.group(3)
        if len(right) == 3 and right.startswith("1"):
            return f"{left}到{right[1:]}{unit}"
        return f"{left}到{right}{unit}"

    cleaned = re.sub(r"(\d{1,2})\s*-\s*(\d{3})(分钟|小时|天|周|次)", _fix_malformed_range, cleaned)
    cleaned = re.sub(r"([，。！？；：、])(?=[A-Za-z0-9])", r"\1 ", cleaned)
    cleaned = re.sub(r"(?<=[A-Za-z0-9])([，。！？；：、])", r" \1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"(?<![。！？])$", "。", cleaned)
    return cleaned.strip()


def count_speech_chars(text: str) -> int:
    cleaned = re.sub(r"[\s，。！？；：、】【（）()、“”\"'…,.!?;:-]", "", str(text or ""))
    return len(cleaned)


def tighten_speech_script(text: str, max_chars: int = 55, max_sentences: int = 2) -> str:
    cleaned = normalize_generated_reply(text).replace("\n", "")
    if not cleaned:
        return "我在这里陪着你。先慢慢呼吸，我们一步一步来。"

    sentence_parts = [part.strip() for part in re.split(r"(?<=[。！？])", cleaned) if part.strip()]
    selected: list[str] = []

    for part in sentence_parts:
        candidate = "".join(selected + [part])
        if len(selected) >= max_sentences:
            break
        if count_speech_chars(candidate) <= max_chars:
            selected.append(part)
        elif not selected:
            selected.append(part)
            break
        else:
            break

    script = "".join(selected) if selected else cleaned

    if count_speech_chars(script) > max_chars:
        compact = re.sub(r"[，；：、】【（）()“”\"'…]", "", script)
        compact_chars: list[str] = []
        counted = 0
        for char in compact:
            if char.isspace():
                continue
            char_cost = 0 if char in "。！？" else 1
            if counted + char_cost > max_chars:
                break
            compact_chars.append(char)
            counted += char_cost
        script = "".join(compact_chars).rstrip("，；：、 ")
        if script and script[-1] not in "。！？":
            script += "。"

    sentence_count = len(re.findall(r"[。！？]", script))
    if sentence_count > max_sentences:
        fragments = [part.strip() for part in re.split(r"(?<=[。！？])", script) if part.strip()]
        script = "".join(fragments[:max_sentences])

    script = script.strip()
    if not script:
        script = "我在这里陪着你。先慢慢呼吸，我们一步一步来。"
    if script[-1] not in "。！？":
        script += "。"
    return script


async def build_speech_script(
    full_reply: str,
    user_input: str,
    top_emotions: list[str],
    graph_grounding: str,
    max_chars: int = 55,
) -> str:
    normalized_reply = normalize_generated_reply(full_reply)
    if count_speech_chars(normalized_reply) <= max_chars and len(re.findall(r"[。！？]", normalized_reply)) <= 2:
        return normalized_reply

    speech_prompt = [
        {
            "role": "system",
            "content": (
                "你是一位温和、专业的心理陪伴助手。"
                "请把长回答压缩成适合8到10秒中文口播的脚本。"
                "要求：只保留最关键的共情和1条建议；只用2句话；不超过55个汉字；"
                "自然口语化，不要编号，不要解释背景，不要扩写。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"用户情绪：{', '.join(top_emotions)}\n"
                f"用户原话：{user_input}\n"
                f"图谱命中摘要：{graph_grounding}\n"
                f"完整回答：{normalized_reply}\n\n"
                "请输出最终口播稿。"
            ),
        },
    ]

    try:
        short_reply = await call_siliconflow_api(speech_prompt, max_tokens=120)
        return tighten_speech_script(short_reply, max_chars=max_chars, max_sentences=2)
    except Exception:
        traceback.print_exc()
        return tighten_speech_script(normalized_reply, max_chars=max_chars, max_sentences=2)


def normalize_emotion_percentages(items: list[dict[str, Any]], decimals: int = 1) -> list[dict[str, float | str]]:
    if not items:
        return [{"name": "平静", "value": 100.0}]

    sanitized = []
    for item in items:
        try:
            raw_value = max(0.0, float(item.get("value", 0.0)))
        except Exception:
            raw_value = 0.0
        sanitized.append({"name": str(item.get("name", "")), "value": raw_value})

    total = sum(item["value"] for item in sanitized)
    if total <= 0:
        return [{"name": "平静", "value": 100.0}]

    scale = 10 ** decimals
    raw_scaled = [(item["value"] / total) * 100 * scale for item in sanitized]
    floors = [math.floor(value) for value in raw_scaled]
    remainder = int(round(100 * scale - sum(floors)))

    ranked_remainders = sorted(
        enumerate(raw_scaled),
        key=lambda pair: pair[1] - floors[pair[0]],
        reverse=True,
    )
    for index, _ in ranked_remainders[: max(0, remainder)]:
        floors[index] += 1

    normalized: list[dict[str, float | str]] = []
    for item, scaled_value in zip(sanitized, floors):
        normalized.append({"name": item["name"], "value": scaled_value / scale})
    return normalized


def load_graph_snapshot() -> dict[str, Any]:
    graphml_path = Path(GRAPHML_PATH)
    if not graphml_path.exists():
        return {"nodes": [], "links": [], "node_lookup": {}, "edge_lookup": {}}

    mtime = graphml_path.stat().st_mtime
    if _graph_snapshot_cache["snapshot"] is not None and _graph_snapshot_cache["mtime"] == mtime:
        return _graph_snapshot_cache["snapshot"]

    graph = nx.read_graphml(graphml_path)
    nodes = [
        {
            "id": str(node_id),
            "name": str(node_id),
            "category": str(data.get("entity_type", "未知实体")),
            "desc": str(data.get("description", "")),
            "val": 1,
        }
        for node_id, data in graph.nodes(data=True)
    ]
    links = [
        {
            "id": f"edge-{index}",
            "source": str(source),
            "target": str(target),
            "label": str(data.get("weight", "") or data.get("description", "")),
        }
        for index, (source, target, data) in enumerate(graph.edges(data=True))
    ]

    node_lookup = {normalize_text(node["name"]): node for node in nodes}
    edge_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for link in links:
        key = tuple(sorted((normalize_text(link["source"]), normalize_text(link["target"]))))
        edge_lookup[key] = link

    snapshot = {
        "nodes": nodes,
        "links": links,
        "node_lookup": node_lookup,
        "edge_lookup": edge_lookup,
    }
    _graph_snapshot_cache["mtime"] = mtime
    _graph_snapshot_cache["snapshot"] = snapshot
    return snapshot


def expand_graph_term_candidates(value: str) -> list[str]:
    raw = str(value or "").strip()
    if not raw:
        return []

    candidates = {raw}
    normalized = normalize_text(raw)
    if normalized:
        candidates.add(normalized)
        if normalized.endswith("s") and len(normalized) > 3:
            candidates.add(normalized[:-1])
        for alias in GRAPH_TERM_ALIASES.get(normalized, []):
            candidates.add(alias)

    words = re.split(r"[\s,_/\-]+", raw)
    for word in words:
        word = word.strip()
        if not word:
            continue
        candidates.add(word)
        word_norm = normalize_text(word)
        if word_norm.endswith("s") and len(word_norm) > 3:
            candidates.add(word_norm[:-1])
        for alias in GRAPH_TERM_ALIASES.get(word_norm, []):
            candidates.add(alias)

    return [candidate for candidate in candidates if str(candidate).strip()]


def score_graph_node_matches(term: str, snapshot: dict[str, Any], knowledge_text: str = "", limit: int = 6) -> list[dict[str, Any]]:
    candidates = expand_graph_term_candidates(term)
    if not candidates:
        return []

    scored: dict[str, tuple[float, dict[str, Any]]] = {}
    knowledge_norm = normalize_text(knowledge_text)

    for candidate in candidates:
        candidate_norm = normalize_text(candidate)
        if not candidate_norm:
            continue
        for node in snapshot["nodes"]:
            node_norm = normalize_text(node["name"])
            if not node_norm:
                continue
            score = 0.0
            if candidate_norm == node_norm:
                score = 120.0
            elif candidate_norm in node_norm:
                score = 88.0
            elif node_norm in candidate_norm and len(node_norm) >= 2:
                score = 76.0
            elif candidate_norm in normalize_text(node.get("desc", "")):
                score = 52.0

            if score <= 0:
                continue
            if knowledge_norm and node_norm in knowledge_norm:
                score += 18.0

            best = scored.get(node["id"])
            if best is None or score > best[0]:
                scored[node["id"]] = (score, node)

    ranked = sorted(scored.values(), key=lambda item: item[0], reverse=True)
    return [node for _, node in ranked[:limit]]


def derive_graph_highlights(
    raw_highlight_nodes: list[str],
    raw_highlight_edges: list[dict[str, str]],
    knowledge_text: str,
    limit_nodes: int = 15,
    limit_edges: int = 18,
) -> tuple[list[str], list[dict[str, str]]]:
    snapshot = load_graph_snapshot()
    if not snapshot["nodes"]:
        return raw_highlight_nodes, raw_highlight_edges

    scored_nodes: dict[str, tuple[float, dict[str, Any]]] = {}
    knowledge_text = str(knowledge_text or "")
    knowledge_norm = normalize_text(knowledge_text)

    def put_node(node: dict[str, Any], score: float) -> None:
        current = scored_nodes.get(node["id"])
        if current is None or score > current[0]:
            scored_nodes[node["id"]] = (score, node)

    for raw in raw_highlight_nodes:
        matches = score_graph_node_matches(raw, snapshot, knowledge_text=knowledge_text, limit=4)
        for rank, node in enumerate(matches):
            put_node(node, 120.0 - rank * 10.0)

    for node in snapshot["nodes"]:
        node_name = str(node["name"])
        node_norm = normalize_text(node_name)
        if len(node_name.strip()) < 2:
            continue
        if knowledge_norm and node_norm and node_norm in knowledge_norm:
            occurrences = knowledge_text.count(node_name)
            put_node(node, 60.0 + occurrences * 4.0)

    selected_nodes = [
        node for _, node in sorted(scored_nodes.values(), key=lambda item: item[0], reverse=True)[:limit_nodes]
    ]

    if not selected_nodes:
        return raw_highlight_nodes, raw_highlight_edges

    selected_norms = {normalize_text(node["name"]) for node in selected_nodes}
    selected_names = [str(node["name"]) for node in selected_nodes]
    selected_node_lookup = {normalize_text(node["name"]): node for node in selected_nodes}
    scored_edges: dict[tuple[str, str], tuple[float, dict[str, str]]] = {}

    def put_edge(source_name: str, target_name: str, score: float) -> None:
        key = tuple(sorted((normalize_text(source_name), normalize_text(target_name))))
        if not key[0] or not key[1] or key[0] == key[1]:
            return
        payload = {"source": source_name, "target": target_name, "raw": f"{source_name} -> {target_name}"}
        current = scored_edges.get(key)
        if current is None or score > current[0]:
            scored_edges[key] = (score, payload)

    for hint in raw_highlight_edges:
        if hint.get("source") and hint.get("target"):
            source_matches = score_graph_node_matches(hint["source"], snapshot, knowledge_text=knowledge_text, limit=2)
            target_matches = score_graph_node_matches(hint["target"], snapshot, knowledge_text=knowledge_text, limit=2)
            for source_node in source_matches:
                for target_node in target_matches:
                    edge_key = tuple(sorted((normalize_text(source_node["name"]), normalize_text(target_node["name"]))))
                    if edge_key in snapshot["edge_lookup"]:
                        put_edge(str(source_node["name"]), str(target_node["name"]), 120.0)

    for link in snapshot["links"]:
        source_norm = normalize_text(link["source"])
        target_norm = normalize_text(link["target"])
        if source_norm not in selected_norms or target_norm not in selected_norms:
            continue
        score = 48.0
        label_norm = normalize_text(link.get("label", ""))
        if knowledge_norm and label_norm and label_norm in knowledge_norm:
            score += 18.0
        if knowledge_norm and source_norm in knowledge_norm and target_norm in knowledge_norm:
            score += 12.0
        put_edge(str(link["source"]), str(link["target"]), score)

    selected_edges = [
        edge for _, edge in sorted(scored_edges.values(), key=lambda item: item[0], reverse=True)[:limit_edges]
    ]

    if not selected_edges and len(selected_names) >= 2:
        center_name = selected_names[0]
        center_norm = normalize_text(center_name)
        for node_name in selected_names[1:]:
            edge_key = tuple(sorted((center_norm, normalize_text(node_name))))
            if edge_key in snapshot["edge_lookup"]:
                put_edge(center_name, node_name, 40.0)
        selected_edges = [
            edge for _, edge in sorted(scored_edges.values(), key=lambda item: item[0], reverse=True)[:limit_edges]
        ]

    # If an edge introduces a node not yet in the selected set, include it.
    for edge in selected_edges:
        for endpoint in (edge["source"], edge["target"]):
            endpoint_norm = normalize_text(endpoint)
            if endpoint_norm not in selected_node_lookup and endpoint_norm in snapshot["node_lookup"]:
                selected_names.append(str(snapshot["node_lookup"][endpoint_norm]["name"]))
                selected_node_lookup[endpoint_norm] = snapshot["node_lookup"][endpoint_norm]
                if len(selected_names) >= limit_nodes:
                    break

    return selected_names[:limit_nodes], selected_edges[:limit_edges]


def build_graph_grounding_brief(
    highlight_nodes: list[str],
    highlight_edges: list[dict[str, str]],
    knowledge_text: str,
    max_nodes: int = 8,
    max_edges: int = 6,
) -> str:
    brief_lines: list[str] = []

    unique_nodes: list[str] = []
    seen_nodes: set[str] = set()
    for node in highlight_nodes:
        cleaned = str(node).strip()
        norm = normalize_text(cleaned)
        if not cleaned or not norm or norm in seen_nodes:
            continue
        seen_nodes.add(norm)
        unique_nodes.append(cleaned)
    if unique_nodes:
        brief_lines.append(f"图谱命中节点：{'、'.join(unique_nodes[:max_nodes])}")

    unique_edges: list[str] = []
    seen_edges: set[str] = set()
    for edge in highlight_edges:
        source = str(edge.get('source', '')).strip()
        target = str(edge.get('target', '')).strip()
        if source and target:
            rendered = f"{source} -> {target}"
            edge_key = tuple(sorted((normalize_text(source), normalize_text(target))))
            dedupe_key = f"{edge_key[0]}__{edge_key[1]}"
        else:
            rendered = str(edge.get("raw") or edge.get("label") or "").strip()
            dedupe_key = normalize_text(rendered)
        if not rendered or not dedupe_key or dedupe_key in seen_edges:
            continue
        seen_edges.add(dedupe_key)
        unique_edges.append(rendered)
    if unique_edges:
        brief_lines.append(f"图谱关系线索：{'；'.join(unique_edges[:max_edges])}")

    knowledge_excerpt = stringify_knowledge(knowledge_text).strip()
    if knowledge_excerpt and knowledge_excerpt != "暂无相关图谱资料。":
        concise_excerpt = knowledge_excerpt[:260]
        if len(knowledge_excerpt) > 260:
            concise_excerpt += "..."
        brief_lines.append(f"检索摘要：{concise_excerpt}")

    return "\n".join(brief_lines) or "本轮没有形成稳定的图谱命中，请明确说明资料有限。"


def resolve_python_executable(preferred_python: str | None = None) -> str:
    if preferred_python:
        candidate = Path(preferred_python).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
    return sys.executable


def build_process_env(preferred_python: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    python_executable = Path(resolve_python_executable(preferred_python))
    python_dir = python_executable.parent
    path_candidates = [
        str(python_dir),
        str(python_dir / "Scripts"),
        str(python_dir / "Library" / "bin"),
        str(python_dir.parent / "Library" / "bin"),
        env.get("PATH", ""),
    ]
    env["PATH"] = os.pathsep.join([item for item in path_candidates if item])
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def resolve_ffmpeg_executable(preferred_python: str | None = None) -> str | None:
    python_executable = Path(resolve_python_executable(preferred_python))
    candidates = [
        python_executable.parent / "ffmpeg",
        python_executable.parent / "Library" / "bin" / "ffmpeg.exe",
        python_executable.parent / "Scripts" / "ffmpeg.exe",
        python_executable.parent / "ffmpeg.exe",
        python_executable.parent.parent / "Library" / "bin" / "ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return shutil.which("ffmpeg")


def resolve_ffprobe_executable(preferred_python: str | None = None) -> str | None:
    python_executable = Path(resolve_python_executable(preferred_python))
    candidates = [
        python_executable.parent / "ffprobe",
        python_executable.parent / "Library" / "bin" / "ffprobe.exe",
        python_executable.parent / "Scripts" / "ffprobe.exe",
        python_executable.parent / "ffprobe.exe",
        python_executable.parent.parent / "Library" / "bin" / "ffprobe.exe",
    ]
    ffmpeg_executable = resolve_ffmpeg_executable(preferred_python)
    if ffmpeg_executable:
        ffmpeg_path = Path(ffmpeg_executable)
        if ffmpeg_path.name.lower().startswith("ffmpeg"):
            candidates.append(ffmpeg_path.with_name(ffmpeg_path.name.replace("ffmpeg", "ffprobe", 1)))
            candidates.append(ffmpeg_path.with_name("ffprobe"))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return shutil.which("ffprobe")


def parse_size_pair(raw_value: str, default: tuple[int, int]) -> tuple[int, int]:
    parts = [item for item in re.split(r"[\s,;xX]+", raw_value.strip()) if item]
    if len(parts) < 2:
        return default
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        return default
    if width <= 0 or height <= 0:
        return default
    return width, height


def parse_edge_hints(raw_groups: list[str], limit: int = 8) -> list[dict[str, str]]:
    pair_counter: Counter[tuple[str, str]] = Counter()
    pair_display: dict[tuple[str, str], tuple[str, str]] = {}
    label_counter: Counter[str] = Counter()
    label_display: dict[str, str] = {}

    pair_patterns = [
        re.compile(r"[\(\[]\s*([^,\]\)]+?)\s*[,，]\s*([^,\]\)]+?)\s*[\)\]]"),
        re.compile(r"(.+?)\s*(?:->|=>|--|—|→|↔|<->|<=>|到|至|to)\s*(.+)"),
    ]

    def add_pair(left: str, right: str) -> None:
        left_clean = left.strip().strip("'\"")
        right_clean = right.strip().strip("'\"")
        left_norm = normalize_text(left_clean)
        right_norm = normalize_text(right_clean)
        if not left_norm or not right_norm:
            return
        key = tuple(sorted((left_norm, right_norm)))
        pair_counter[key] += 1
        pair_display.setdefault(key, (left_clean, right_clean))

    def add_label(label: str) -> None:
        cleaned = label.strip().strip("[](){}'\"")
        norm = normalize_text(cleaned)
        if not norm:
            return
        label_counter[norm] += 1
        label_display.setdefault(norm, cleaned)

    for group in raw_groups:
        fragment = str(group).strip()
        if not fragment:
            continue

        found_pair = False
        for pattern in pair_patterns:
            matches = pattern.findall(fragment)
            if not matches:
                continue
            for left, right in matches:
                add_pair(left, right)
                found_pair = True

        if found_pair:
            continue

        for part in re.split(r"[;,；]", fragment):
            candidate = part.strip()
            if not candidate:
                continue
            matched_pair = False
            for pattern in pair_patterns[1:]:
                matched = pattern.fullmatch(candidate)
                if matched:
                    add_pair(matched.group(1), matched.group(2))
                    matched_pair = True
                    break
            if not matched_pair:
                add_label(candidate)

    edge_hints: list[dict[str, str]] = []
    for key, _ in pair_counter.most_common(limit):
        left, right = pair_display[key]
        edge_hints.append({"source": left, "target": right, "raw": f"{left} -> {right}"})
    remaining = max(0, limit - len(edge_hints))
    for label_key, _ in label_counter.most_common(remaining):
        edge_hints.append({"label": label_display[label_key], "raw": label_display[label_key]})
    return edge_hints


def resolve_ditto_dir() -> Path | None:
    candidates = []
    if DITTO_DIR:
        candidates.append(Path(DITTO_DIR).expanduser())
    candidates.extend(
        [
            Path.cwd() / "ditto-talkinghead",
            Path.cwd().parent / "ditto-talkinghead",
            Path.cwd() / "LivePortrait",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and (candidate / "inference.py").exists():
            return candidate.resolve()
    return None


def resolve_echomimic_v3_dir() -> Path | None:
    candidates = []
    if ECHOMIMIC_V3_DIR:
        candidates.append(Path(ECHOMIMIC_V3_DIR).expanduser())
    candidates.extend(
        [
            Path.cwd() / "EchoMimicV3",
            Path.cwd() / "echomimic_v3",
            Path.cwd().parent / "EchoMimicV3",
            Path.cwd().parent / "echomimic_v3",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and (candidate / "infer_flash.py").exists():
            return candidate.resolve()
    return None


def resolve_echomimic_v3_python() -> str | None:
    candidates = []
    if ECHOMIMIC_V3_PYTHON:
        candidates.append(Path(ECHOMIMIC_V3_PYTHON).expanduser())
    candidates.extend(
        [
            Path("/root/miniconda3/envs/echomimic_v3/bin/python"),
            Path("/opt/conda/envs/echomimic_v3/bin/python"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def resolve_echomimic_v3_script(echomimic_dir: Path) -> Path | None:
    if ECHOMIMIC_V3_SCRIPT:
        script_path = Path(ECHOMIMIC_V3_SCRIPT).expanduser()
        if not script_path.is_absolute():
            script_path = echomimic_dir / script_path
        if script_path.exists():
            return script_path.resolve()
    candidate = echomimic_dir / "infer_flash.py"
    return candidate.resolve() if candidate.exists() else None


def resolve_echomimic_v3_config(echomimic_dir: Path) -> Path | None:
    candidates = []
    if ECHOMIMIC_V3_CONFIG:
        candidates.append(Path(ECHOMIMIC_V3_CONFIG).expanduser())
    candidates.extend(
        [
            echomimic_dir / "config" / "config.yaml",
            echomimic_dir / "config" / "wan2.1" / "wan_civitai.yaml",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_echomimic_v3_bundle_root(echomimic_dir: Path) -> Path | None:
    candidates = []
    if ECHOMIMIC_V3_MODEL_ROOT:
        candidates.append(Path(ECHOMIMIC_V3_MODEL_ROOT).expanduser())
    candidates.extend(
        [
            echomimic_dir / "flash-pro",
            echomimic_dir / "flash",
            echomimic_dir / "weights" / "flash-pro",
            echomimic_dir / "weights" / "flash",
            echomimic_dir.parent / "flash-pro",
            echomimic_dir.parent / "flash",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_echomimic_v3_model_name(bundle_root: Path) -> Path | None:
    candidates = [bundle_root / "Wan2.1-Fun-V1.1-1.3B-InP"]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_echomimic_v3_transformer(bundle_root: Path) -> Path | None:
    candidates = []
    if ECHOMIMIC_V3_TRANSFORMER_PATH:
        candidates.append(Path(ECHOMIMIC_V3_TRANSFORMER_PATH).expanduser())
    candidates.append(bundle_root / "transformer" / "diffusion_pytorch_model.safetensors")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_echomimic_v3_wav2vec(bundle_root: Path) -> Path | None:
    candidates = []
    if ECHOMIMIC_V3_WAV2VEC_DIR:
        candidates.append(Path(ECHOMIMIC_V3_WAV2VEC_DIR).expanduser())
    candidates.extend(
        [
            bundle_root / "chinese-wav2vec2-base",
            bundle_root / "wav2vec2-base-960h",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_wav2lip_dir() -> Path | None:
    candidates = []
    if WAV2LIP_DIR:
        candidates.append(Path(WAV2LIP_DIR).expanduser())
    candidates.extend(
        [
            Path.cwd() / "Wav2Lip",
            Path.cwd() / "wav2lip",
            Path.cwd().parent / "Wav2Lip",
            Path.cwd().parent / "wav2lip",
        ]
    )
    for candidate in candidates:
        if candidate.exists() and (candidate / "inference.py").exists():
            return candidate.resolve()
    return None


def resolve_wav2lip_checkpoint(wav2lip_dir: Path) -> Path | None:
    candidates = []
    if WAV2LIP_CHECKPOINT:
        candidates.append(Path(WAV2LIP_CHECKPOINT).expanduser())
    candidates.extend(
        [
            wav2lip_dir / "checkpoints" / "wav2lip.pth",
            wav2lip_dir / "checkpoints" / "wav2lip_gan.pth",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_wav2lip_face_detector(wav2lip_dir: Path) -> Path | None:
    candidates = []
    if WAV2LIP_FACE_DET:
        candidates.append(Path(WAV2LIP_FACE_DET).expanduser())
    candidates.append(wav2lip_dir / "face_detection" / "detection" / "sfd" / "s3fd.pth")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_wav2lip_command(wav2lip_dir: Path, checkpoint_path: Path, audio_path: str, output_path: str) -> list[str]:
    source_image_path = str(Path(SOURCE_IMAGE).resolve())
    audio_file_path = str(Path(audio_path).resolve())
    output_file_path = str(Path(output_path).resolve())
    command = [
        resolve_python_executable(WAV2LIP_PYTHON),
        str(wav2lip_dir / "inference.py"),
        "--checkpoint_path",
        str(checkpoint_path),
        "--face",
        source_image_path,
        "--audio",
        audio_file_path,
        "--outfile",
        output_file_path,
        "--fps",
        str(WAV2LIP_FPS),
        "--face_det_batch_size",
        str(WAV2LIP_FACE_DET_BATCH_SIZE),
        "--wav2lip_batch_size",
        str(WAV2LIP_BATCH_SIZE),
        "--resize_factor",
        str(WAV2LIP_RESIZE_FACTOR),
        "--nosmooth",
    ]
    pads = [item for item in WAV2LIP_PADS.split() if item]
    if len(pads) == 4:
        command.extend(["--pads", *pads])
    return command


def write_debug_log(name: str, content: str) -> None:
    try:
        log_path = Path(MEDIA_DIR) / name
        log_path.write_text(content, encoding="utf-8", errors="ignore")
    except Exception:
        pass


def convert_audio_to_wav(audio_path: str, wav_output_path: str, preferred_python: str | None = None) -> str:
    ffmpeg_executable = resolve_ffmpeg_executable(preferred_python)
    if not ffmpeg_executable:
        raise RuntimeError("ffmpeg executable not found for Wav2Lip audio preprocessing.")

    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(Path(audio_path).resolve()),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(Path(wav_output_path).resolve()),
    ]
    process_env = build_process_env(preferred_python)
    result = subprocess.run(command, check=True, capture_output=True, text=True, env=process_env)
    if result.stdout:
        print(result.stdout[-800:])
    if result.stderr:
        print(result.stderr[-800:])
    return str(Path(wav_output_path).resolve())


def probe_audio_duration_seconds(audio_path: str, preferred_python: str | None = None) -> float | None:
    ffprobe_executable = resolve_ffprobe_executable(preferred_python)
    if ffprobe_executable:
        command = [
            ffprobe_executable,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(Path(audio_path).resolve()),
        ]
        process_env = build_process_env(preferred_python)
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, env=process_env)
            return float((result.stdout or "").strip())
        except Exception:
            pass

    try:
        import torchaudio

        audio_meta = torchaudio.info(str(Path(audio_path).resolve()))
        if audio_meta.sample_rate and audio_meta.num_frames:
            return float(audio_meta.num_frames) / float(audio_meta.sample_rate)
    except Exception:
        pass

    try:
        import soundfile as sf

        audio_meta = sf.info(str(Path(audio_path).resolve()))
        if audio_meta.samplerate and audio_meta.frames:
            return float(audio_meta.frames) / float(audio_meta.samplerate)
    except Exception:
        pass

    return None


def build_echomimic_v3_command(
    script_path: Path,
    config_path: Path,
    model_name_path: Path,
    transformer_path: Path,
    wav2vec_dir: Path,
    audio_path: str,
    save_dir: Path,
    memory_mode_override: str | None = None,
    partial_video_length_override: int | None = None,
    overlap_video_length_override: int | None = None,
) -> list[str]:
    source_image_path = str(Path(SOURCE_IMAGE).resolve())
    audio_file_path = str(Path(audio_path).resolve())
    sample_width, sample_height = parse_size_pair(ECHOMIMIC_V3_SAMPLE_SIZE, default=(768, 768))

    duration_seconds = probe_audio_duration_seconds(audio_file_path, resolve_echomimic_v3_python()) or 0.0
    estimated_frames = max(1, int(duration_seconds * ECHOMIMIC_V3_FPS + 0.999))
    video_length = max(1, estimated_frames)
    if ECHOMIMIC_V3_MAX_FRAMES > 0:
        video_length = min(video_length, ECHOMIMIC_V3_MAX_FRAMES)

    partial_video_length = max(1, int(partial_video_length_override or ECHOMIMIC_V3_PARTIAL_VIDEO_LENGTH))
    overlap_video_length = max(0, int(overlap_video_length_override if overlap_video_length_override is not None else ECHOMIMIC_V3_OVERLAP_VIDEO_LENGTH))
    if overlap_video_length >= partial_video_length:
        overlap_video_length = max(0, partial_video_length - 1)
    memory_mode = memory_mode_override or ECHOMIMIC_V3_GPU_MEMORY_MODE

    command = [
        resolve_python_executable(resolve_echomimic_v3_python()),
        str(script_path),
        "--image_path",
        source_image_path,
        "--audio_path",
        audio_file_path,
        "--prompt",
        ECHOMIMIC_V3_PROMPT,
        "--num_inference_steps",
        str(ECHOMIMIC_V3_STEPS),
        "--config_path",
        str(config_path),
        "--model_name",
        str(model_name_path),
        "--transformer_path",
        str(transformer_path),
        "--save_path",
        str(save_dir.resolve()),
        "--wav2vec_model_dir",
        str(wav2vec_dir),
        "--sampler_name",
        "Flow_Unipc",
        "--video_length",
        str(video_length),
        "--partial_video_length",
        str(partial_video_length),
        "--overlap_video_length",
        str(overlap_video_length),
        "--guidance_scale",
        str(ECHOMIMIC_V3_GUIDANCE_SCALE),
        "--audio_guidance_scale",
        str(ECHOMIMIC_V3_AUDIO_GUIDANCE_SCALE),
        "--audio_scale",
        str(ECHOMIMIC_V3_AUDIO_SCALE),
        "--neg_scale",
        "1.0",
        "--neg_steps",
        "0",
        "--seed",
        "43",
        "--enable_teacache",
        "--teacache_threshold",
        "0.1",
        "--num_skip_start_steps",
        "5",
        "--riflex_k",
        "6",
        "--ulysses_degree",
        "1",
        "--ring_degree",
        "1",
        "--weight_dtype",
        ECHOMIMIC_V3_WEIGHT_DTYPE,
        "--GPU_memory_mode",
        memory_mode,
        "--sample_size",
        str(sample_width),
        str(sample_height),
        "--fps",
        str(ECHOMIMIC_V3_FPS),
        "--add_prompt",
        "",
        "--negative_prompt",
        "",
        "--shift",
        str(ECHOMIMIC_V3_SHIFT),
    ]
    if ECHOMIMIC_V3_EXTRA_ARGS:
        command.extend(shlex.split(ECHOMIMIC_V3_EXTRA_ARGS, posix=False))
    return command


def output_has_cuda_oom(output: str) -> bool:
    normalized = str(output or "").lower()
    return "cuda out of memory" in normalized or "torch.outofmemoryerror" in normalized


def run_echomimic_attempt(
    command: list[str],
    echomimic_dir: Path,
    process_env: dict[str, str],
    log_name: str,
) -> tuple[bool, str]:
    print(f"[EchoMimicV3] command: {' '.join(command)}")
    return_code, combined_output = run_process_with_terminal_progress(
        command,
        cwd=echomimic_dir,
        env=process_env,
        prefix="[EchoMimicV3]",
    )
    if return_code == 0:
        return True, combined_output

    print(f"[EchoMimicV3] 运行失败，退出码: {return_code}")
    if combined_output:
        print(combined_output[-1600:])
    write_debug_log(
        log_name,
        f"COMMAND:\n{' '.join(command)}\n\nOUTPUT:\n{combined_output or ''}\n",
    )
    return False, combined_output


def generate_echomimic_v3_video(audio_path: str, output_path: str) -> str | None:
    echomimic_dir = resolve_echomimic_v3_dir()
    if not echomimic_dir:
        return None

    script_path = resolve_echomimic_v3_script(echomimic_dir)
    if not script_path:
        print(f"[EchoMimicV3] 在 {echomimic_dir} 下未找到 infer_flash.py。")
        return None

    config_path = resolve_echomimic_v3_config(echomimic_dir)
    if not config_path:
        print("[EchoMimicV3] 未找到 config/config.yaml。")
        return None

    bundle_root = resolve_echomimic_v3_bundle_root(echomimic_dir)
    if not bundle_root:
        print("[EchoMimicV3] 未找到 flash-pro 权重目录。")
        return None

    model_name_path = resolve_echomimic_v3_model_name(bundle_root)
    transformer_path = resolve_echomimic_v3_transformer(bundle_root)
    wav2vec_dir = resolve_echomimic_v3_wav2vec(bundle_root)
    if not model_name_path or not transformer_path or not wav2vec_dir:
        print("[EchoMimicV3] 模型目录不完整，至少需要 base model / transformer / chinese-wav2vec2-base。")
        return None

    save_dir = Path(output_path).resolve().parent / f"{Path(output_path).stem}_echomimic_v3"
    save_dir.mkdir(parents=True, exist_ok=True)

    command = build_echomimic_v3_command(
        script_path=script_path,
        config_path=config_path,
        model_name_path=model_name_path,
        transformer_path=transformer_path,
        wav2vec_dir=wav2vec_dir,
        audio_path=audio_path,
        save_dir=save_dir,
    )
    process_env = build_process_env(resolve_echomimic_v3_python())
    process_env["PYTHONPATH"] = str(echomimic_dir) + os.pathsep + process_env.get("PYTHONPATH", "")

    start_time = time.time()
    try:
        success, combined_output = run_echomimic_attempt(
            command,
            echomimic_dir=echomimic_dir,
            process_env=process_env,
            log_name="echomimic_v3_last_error.log",
        )
        if not success and output_has_cuda_oom(combined_output):
            fallback_partial = min(49, max(33, ECHOMIMIC_V3_PARTIAL_VIDEO_LENGTH - 16))
            fallback_overlap = min(4, max(0, fallback_partial - 1))
            fallback_command = build_echomimic_v3_command(
                script_path=script_path,
                config_path=config_path,
                model_name_path=model_name_path,
                transformer_path=transformer_path,
                wav2vec_dir=wav2vec_dir,
                audio_path=audio_path,
                save_dir=save_dir,
                memory_mode_override="sequential_cpu_offload",
                partial_video_length_override=fallback_partial,
                overlap_video_length_override=fallback_overlap,
            )
            print(
                "[EchoMimicV3] 检测到 CUDA OOM，自动切换到更省显存的重试方案："
                f"GPU_memory_mode=sequential_cpu_offload, partial_video_length={fallback_partial}, "
                f"overlap_video_length={fallback_overlap}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            success, combined_output = run_echomimic_attempt(
                fallback_command,
                echomimic_dir=echomimic_dir,
                process_env=process_env,
                log_name="echomimic_v3_last_retry_error.log",
            )
        if not success:
            return None
    except Exception as exc:
        print(f"[EchoMimicV3] 调用异常: {exc}")
        write_debug_log(
            "echomimic_v3_last_error.log",
            f"COMMAND:\n{' '.join(command)}\n\nEXCEPTION:\n{exc}\n",
        )
        return None

    resolved_output = locate_generated_video(Path(output_path), save_dir, start_time)
    if not resolved_output:
        print("[EchoMimicV3] 推理完成，但未检测到输出视频。")
        write_debug_log(
            "echomimic_v3_last_error.log",
            f"COMMAND:\n{' '.join(command)}\n\nOUTPUT:\n{combined_output or ''}\n\nERROR:\nNo output video detected.\n",
        )
        return None
    return str(resolved_output)


def resolve_ditto_script(ditto_dir: Path) -> Path | None:
    if DITTO_SCRIPT:
        script_path = Path(DITTO_SCRIPT).expanduser()
        if not script_path.is_absolute():
            script_path = ditto_dir / script_path
        if script_path.exists():
            return script_path.resolve()

    candidate = ditto_dir / "inference.py"
    return candidate.resolve() if candidate.exists() else None


def resolve_ditto_checkpoint_root(ditto_dir: Path) -> Path | None:
    candidates = []
    if DITTO_CHECKPOINT_ROOT:
        candidates.append(Path(DITTO_CHECKPOINT_ROOT).expanduser())
    candidates.extend(
        [
            ditto_dir / "checkpoints",
            ditto_dir.parent / "checkpoints",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_ditto_model_root(checkpoint_root: Path) -> Path | None:
    candidates = []
    if DITTO_MODEL_ROOT:
        candidates.append(Path(DITTO_MODEL_ROOT).expanduser())
    candidates.extend(
        [
            checkpoint_root / "ditto_pytorch",
            checkpoint_root / "ditto_trt_Ampere_Plus",
            checkpoint_root / "ditto_trt_custom",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_ditto_cfg_path(checkpoint_root: Path, model_root: Path) -> Path | None:
    candidates = []
    if DITTO_CFG_PKL:
        candidates.append(Path(DITTO_CFG_PKL).expanduser())

    cfg_dir = checkpoint_root / "ditto_cfg"
    if "pytorch" in model_root.name.lower():
        candidates.append(cfg_dir / "v0.4_hubert_cfg_pytorch.pkl")
    candidates.extend(
        [
            cfg_dir / "v0.4_hubert_cfg_trt.pkl",
            cfg_dir / "v0.4_hubert_cfg_trt_online.pkl",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def build_ditto_command(script_path: Path, model_root: Path, cfg_path: Path, audio_path: str, output_path: str) -> list[str]:
    source_image_path = str(Path(SOURCE_IMAGE).resolve())
    audio_file_path = str(Path(audio_path).resolve())
    output_file_path = str(Path(output_path).resolve())
    return [
        resolve_python_executable(DITTO_PYTHON),
        str(script_path),
        "--data_root",
        str(model_root),
        "--cfg_pkl",
        str(cfg_path),
        "--audio_path",
        audio_file_path,
        "--source_path",
        source_image_path,
        "--output_path",
        output_file_path,
    ]


def resolve_liveportrait_dir() -> Path | None:
    candidates = []
    if LIVEPORTRAIT_DIR:
        candidates.append(Path(LIVEPORTRAIT_DIR).expanduser())
    candidates.extend(
        [
            Path.cwd() / "LivePortrait",
            Path.cwd().parent / "LivePortrait",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_liveportrait_script(liveportrait_dir: Path) -> Path | None:
    if LIVEPORTRAIT_SCRIPT:
        script_path = Path(LIVEPORTRAIT_SCRIPT).expanduser()
        if not script_path.is_absolute():
            script_path = liveportrait_dir / script_path
        if script_path.exists():
            return script_path.resolve()

    for relative_path in ("run_audio.py", "run.py", "inference.py"):
        candidate = liveportrait_dir / relative_path
        if candidate.exists():
            return candidate.resolve()
    return None


def build_liveportrait_command(script_path: Path, audio_path: str, output_path: str) -> list[str]:
    source_image_path = str(Path(SOURCE_IMAGE).resolve())
    audio_file_path = str(Path(audio_path).resolve())
    output_file_path = str(Path(output_path).resolve())
    command = [
        sys.executable,
        str(script_path),
        "--source",
        source_image_path,
        "--driving_audio",
        audio_file_path,
        "--output",
        output_file_path,
        "--flag_do_crop",
    ]
    if LIVEPORTRAIT_EXTRA_ARGS:
        command.extend(shlex.split(LIVEPORTRAIT_EXTRA_ARGS, posix=False))
    return command


def locate_generated_video(expected_output: Path, liveportrait_dir: Path, start_time: float) -> Path | None:
    if expected_output.exists():
        return expected_output

    search_roots = [expected_output.parent, liveportrait_dir]
    candidates: list[Path] = []

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in ("*.mp4", "*.mov", "*.webm"):
            for path in root.rglob(pattern):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_mtime >= start_time - 2:
                    candidates.append(path)

    if not candidates:
        return None

    newest = max(candidates, key=lambda item: item.stat().st_mtime)
    if newest.resolve() != expected_output.resolve():
        shutil.copy2(newest, expected_output)
    return expected_output


def generate_wav2lip_video(audio_path: str, output_path: str) -> str | None:
    wav2lip_dir = resolve_wav2lip_dir()
    if not wav2lip_dir:
        return None

    checkpoint_path = resolve_wav2lip_checkpoint(wav2lip_dir)
    if not checkpoint_path:
        print("[Wav2Lip] 未找到 wav2lip.pth 或 wav2lip_gan.pth。")
        return None

    face_detector_path = resolve_wav2lip_face_detector(wav2lip_dir)
    if not face_detector_path:
        print("[Wav2Lip] 未找到 s3fd.pth。")
        return None

    temp_dir = wav2lip_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_audio_path = str((temp_dir / f"{Path(audio_path).stem}_16k.wav").resolve())

    try:
        convert_audio_to_wav(audio_path, wav_audio_path, WAV2LIP_PYTHON)
    except subprocess.CalledProcessError as exc:
        print(f"[Wav2Lip] ffmpeg 音频预处理失败: {exc}")
        write_debug_log(
            "wav2lip_last_error.log",
            f"FFMPEG COMMAND FAILED\nSTDOUT:\n{exc.stdout or ''}\n\nSTDERR:\n{exc.stderr or ''}\n",
        )
        return None
    except Exception as exc:
        print(f"[Wav2Lip] 音频预处理异常: {exc}")
        write_debug_log("wav2lip_last_error.log", f"FFMPEG PREPROCESS EXCEPTION:\n{exc}\n")
        return None

    start_time = time.time()
    command = build_wav2lip_command(wav2lip_dir, checkpoint_path, wav_audio_path, output_path)
    process_env = build_process_env(WAV2LIP_PYTHON)
    process_env["PYTHONPATH"] = str(wav2lip_dir) + os.pathsep + process_env.get("PYTHONPATH", "")
    process_env["WAV2LIP_FACE_DET_PATH"] = str(face_detector_path)

    try:
        print(f"[Wav2Lip] command: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=wav2lip_dir,
            env=process_env,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout[-1200:])
        if result.stderr:
            print(result.stderr[-1200:])
    except subprocess.CalledProcessError as exc:
        print(f"[Wav2Lip] 运行失败: {exc}")
        if exc.stdout:
            print(exc.stdout[-1200:])
        if exc.stderr:
            print(exc.stderr[-1200:])
        write_debug_log(
            "wav2lip_last_error.log",
            f"COMMAND:\n{' '.join(command)}\n\nSTDOUT:\n{exc.stdout or ''}\n\nSTDERR:\n{exc.stderr or ''}\n",
        )
        return None
    except Exception as exc:
        print(f"[Wav2Lip] 调用异常: {exc}")
        write_debug_log("wav2lip_last_error.log", f"COMMAND:\n{' '.join(command)}\n\nEXCEPTION:\n{exc}\n")
        return None

    resolved_output = locate_generated_video(Path(output_path), wav2lip_dir, start_time)
    if not resolved_output:
        print("[Wav2Lip] 推理完成，但未检测到输出视频。")
        write_debug_log(
            "wav2lip_last_error.log",
            f"COMMAND:\n{' '.join(command)}\n\nSTDOUT:\n{result.stdout or ''}\n\nSTDERR:\n{result.stderr or ''}\n\nERROR:\nNo output video detected.\n",
        )
        return None
    return str(resolved_output)


def generate_ditto_video(audio_path: str, output_path: str) -> str | None:
    ditto_dir = resolve_ditto_dir()
    if not ditto_dir:
        return None

    script_path = resolve_ditto_script(ditto_dir)
    if not script_path:
        print(f"[Ditto] 在 {ditto_dir} 下未找到 inference.py。")
        return None

    checkpoint_root = resolve_ditto_checkpoint_root(ditto_dir)
    if not checkpoint_root:
        print(f"[Ditto] 未找到 checkpoints 目录。")
        return None

    model_root = resolve_ditto_model_root(checkpoint_root)
    if not model_root:
        print(f"[Ditto] 未找到可用模型目录（ditto_pytorch / ditto_trt_*）。")
        return None

    cfg_path = resolve_ditto_cfg_path(checkpoint_root, model_root)
    if not cfg_path:
        print(f"[Ditto] 未找到与模型匹配的配置文件。")
        return None

    start_time = time.time()
    command = build_ditto_command(script_path, model_root, cfg_path, audio_path, output_path)

    try:
        result = subprocess.run(
            command,
            cwd=ditto_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout[-1200:])
        if result.stderr:
            print(result.stderr[-1200:])
    except subprocess.CalledProcessError as exc:
        print(f"[Ditto] 运行失败: {exc}")
        if exc.stdout:
            print(exc.stdout[-1200:])
        if exc.stderr:
            print(exc.stderr[-1200:])
        return None
    except Exception as exc:
        print(f"[Ditto] 调用异常: {exc}")
        return None

    resolved_output = locate_generated_video(Path(output_path), ditto_dir, start_time)
    if not resolved_output:
        print("[Ditto] 推理完成，但未检测到输出视频。")
        return None
    return str(resolved_output)


def generate_digital_human_video(audio_path: str, output_path: str) -> str | None:
    echomimic_output = generate_echomimic_v3_video(audio_path, output_path)
    if echomimic_output:
        return echomimic_output

    wav2lip_output = generate_wav2lip_video(audio_path, output_path)
    if wav2lip_output:
        return wav2lip_output

    ditto_output = generate_ditto_video(audio_path, output_path)
    if ditto_output:
        return ditto_output

    liveportrait_dir = resolve_liveportrait_dir()
    if not liveportrait_dir:
        print("[DigitalHuman] 未找到 EchoMimicV3 / Wav2Lip / Ditto / LivePortrait 目录，已退回音频模式。")
        return None

    script_path = resolve_liveportrait_script(liveportrait_dir)
    if not script_path:
        print(f"[LivePortrait] 在 {liveportrait_dir} 下未找到可执行脚本，已退回音频模式。")
        return None

    start_time = time.time()
    command = build_liveportrait_command(script_path, audio_path, output_path)

    try:
        result = subprocess.run(
            command,
            cwd=liveportrait_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout[-1000:])
        if result.stderr:
            print(result.stderr[-1000:])
    except subprocess.CalledProcessError as exc:
        print(f"[LivePortrait] 运行失败: {exc}")
        if exc.stdout:
            print(exc.stdout[-1000:])
        if exc.stderr:
            print(exc.stderr[-1000:])
        return None
    except Exception as exc:
        print(f"[LivePortrait] 调用异常: {exc}")
        return None

    resolved_output = locate_generated_video(Path(output_path), liveportrait_dir, start_time)
    if not resolved_output:
        print("[LivePortrait] 脚本执行完成，但未检测到输出视频文件。")
        return None
    return str(resolved_output)


class RetrieveNodeFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self.last_nodes: list[str] = []
        self.last_edges: list[str] = []

    def reset(self) -> None:
        self.last_nodes = []
        self.last_edges = []

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if "Query nodes:" in msg:
            match = re.search(r"Query nodes:\s*(.*?)\s*\(", msg)
            if match:
                self.last_nodes.append(match.group(1).strip())
        if "Query edges:" in msg:
            match = re.search(r"Query edges:\s*(.*?)\s*\(", msg)
            if match:
                self.last_edges.append(match.group(1).strip())
        return True


node_filter = RetrieveNodeFilter()
logging.getLogger("lightrag").addFilter(node_filter)


async def generate_voice(text: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, "zh-CN-XiaoxiaoNeural")
    await communicate.save(output_path)


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class NoCacheStaticFiles(StaticFiles):
    async def __call__(self, scope, receive, send):
        async def patched_send(message):
            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                headers[b"cache-control"] = b"no-cache, no-store, must-revalidate"
                headers[b"pragma"] = b"no-cache"
                headers[b"expires"] = b"0"
                message["headers"] = list(headers.items())
            await send(message)

        # Wrap original send
        await super().__call__(scope, receive, patched_send)


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

emotion_classifier = None
rag = None
TEXT_EMOTION_MODEL = os.getenv("TEXT_EMOTION_MODEL", "Johnson8187/Chinese-Emotion-Small")
TEXT_EMOTION_CACHE_DIR = os.getenv(
    "TEXT_EMOTION_CACHE_DIR",
    str(PROJECT_ROOT / "models" / "text-emotion"),
)

emotion_map = {
    "admiration": "钦佩",
    "amusement": "有趣",
    "anger": "愤怒",
    "annoyance": "烦躁",
    "approval": "认同",
    "caring": "关心",
    "confusion": "困惑",
    "curiosity": "好奇",
    "desire": "渴望",
    "disappointment": "失望",
    "disapproval": "不认同",
    "disgust": "厌恶",
    "embarrassment": "尴尬",
    "excitement": "兴奋",
    "fear": "恐惧",
    "gratitude": "感激",
    "grief": "悲痛",
    "joy": "喜悦",
    "love": "爱",
    "nervousness": "紧张",
    "optimism": "乐观",
    "pride": "自豪",
    "realization": "领悟",
    "relief": "宽慰",
    "remorse": "懊悔",
    "sadness": "悲伤",
    "surprise": "惊讶",
    "neutral": "平静",
    "neutral tone": "平静",
    "平淡語氣": "平静",
    "平淡语气": "平静",
    "concerned tone": "关心",
    "關切語調": "关心",
    "关切语调": "关心",
    "happy tone": "高兴",
    "開心語調": "高兴",
    "开心语调": "高兴",
    "angry tone": "愤怒",
    "憤怒語調": "愤怒",
    "愤怒语调": "愤怒",
    "sad tone": "悲伤",
    "悲傷語調": "悲伤",
    "悲伤语调": "悲伤",
    "questioning tone": "困惑",
    "疑問語調": "困惑",
    "疑问语调": "困惑",
    "surprised tone": "惊讶",
    "驚奇語調": "惊讶",
    "惊奇语调": "惊讶",
    "disgusted tone": "厌恶",
    "厭惡語調": "厌恶",
    "厌恶语调": "厌恶",
}


def map_text_emotion_label(label: str) -> str:
    raw = str(label or "").strip()
    if not raw:
        return "平静"
    return emotion_map.get(raw, emotion_map.get(raw.lower(), raw))


def local_text_emotion_model_ready(model_name_or_path: str, cache_dir: str) -> bool:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.exists() and (
        (candidate / "model.safetensors").exists() or (candidate / "pytorch_model.bin").exists()
    ):
        return True

    cache_root = Path(cache_dir)
    if not cache_root.exists():
        return False

    if any(cache_root.rglob("model.safetensors")) or any(cache_root.rglob("pytorch_model.bin")):
        return True
    return False


async def analyze_text_emotions_with_llm(text: str) -> list[dict[str, Any]]:
    prompt = [
        {
            "role": "system",
            "content": (
                "你是一个中文情绪识别器。"
                "请根据用户文本输出前5个最相关情绪及百分比。"
                "只输出 JSON 数组，每个元素格式为 "
                '{"name":"情绪","value":数值}。'
                "百分比总和必须约等于100。"
                "情绪名称请使用中文，优先从这些词中选择："
                "平静、紧张、焦虑、悲伤、高兴、愤怒、困惑、失望、关心、惊讶、厌恶、渴望、宽慰、乐观。"
            ),
        },
        {
            "role": "user",
            "content": f"请分析下面这句话的情绪：{text}",
        },
    ]
    raw = await call_siliconflow_api(prompt, max_tokens=180)
    match = re.search(r"\[[\s\S]*\]", raw)
    payload = match.group(0) if match else raw
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise ValueError("LLM emotion output is not a list")
    sanitized = []
    for item in parsed[:5]:
        if not isinstance(item, dict):
            continue
        name = map_text_emotion_label(str(item.get("name", "")).strip())
        try:
            value = float(item.get("value", 0.0))
        except Exception:
            value = 0.0
        if name:
            sanitized.append({"name": name, "value": max(0.0, value)})
    return normalize_emotion_percentages(sanitized or [{"name": "平静", "value": 100.0}], decimals=1)


async def analyze_text_emotions(text: str) -> list[dict[str, Any]]:
    global emotion_classifier
    if emotion_classifier is not None:
        try:
            all_results = emotion_classifier(text)[0]
            all_results.sort(key=lambda item: item["score"], reverse=True)
            raw_emotions = [
                {"name": map_text_emotion_label(item["label"]), "value": float(item["score"]) * 100.0}
                for item in all_results[:5]
            ]
            return normalize_emotion_percentages(raw_emotions, decimals=1)
        except Exception:
            logger.exception("Local text emotion model inference failed; fallback to LLM.")

    return await analyze_text_emotions_with_llm(text)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global emotion_classifier, rag

    try:
        Path(TEXT_EMOTION_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        if local_text_emotion_model_ready(TEXT_EMOTION_MODEL, TEXT_EMOTION_CACHE_DIR):
            text_emotion_tokenizer = AutoTokenizer.from_pretrained(
                TEXT_EMOTION_MODEL,
                cache_dir=TEXT_EMOTION_CACHE_DIR,
                local_files_only=True,
            )
            emotion_classifier = pipeline(
                "text-classification",
                model=TEXT_EMOTION_MODEL,
                tokenizer=text_emotion_tokenizer,
                top_k=None,
                device=0 if torch.cuda.is_available() else -1,
                local_files_only=True,
            )
            logger.info("Loaded text emotion model from %s", TEXT_EMOTION_MODEL)
        else:
            emotion_classifier = None
            logger.warning(
                "Local text emotion model cache not found for %s; use LLM fallback instead of blocking startup.",
                TEXT_EMOTION_MODEL,
            )
    except Exception:
        emotion_classifier = None
        logger.exception("Failed to initialize local text emotion model; fallback to LLM text emotion analysis.")

    rag = kb_rag
    rag.llm_model_func = local_llm_for_rag
    await rag.initialize_storages()
    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app.router.lifespan_context = lifespan


async def call_siliconflow_api(messages: list[dict[str, str]], max_tokens: int = 512) -> str:
    if not SILICONFLOW_API_KEY:
        raise RuntimeError("未配置 SILICONFLOW_API_KEY。")

    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                detail = await response.text()
                raise RuntimeError(f"SiliconFlow API Error {response.status}: {detail}")

            data = await response.json()
            return data["choices"][0]["message"]["content"]


async def local_llm_for_rag(prompt: str, system_prompt: str | None = None, history_messages: list[dict[str, str]] | None = None, **kwargs) -> str:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt + "\nIMPORTANT: Simplified Chinese Output."})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    return await call_siliconflow_api(messages, max_tokens=512)
class ChatRequest(BaseModel):
    message: str
    input_mode: str = "text"
    speech_emotions: list[dict[str, Any]] | None = None
    speech_emotion_model: str | None = None


class SpeechEmotionResponse(BaseModel):
    emotions: list[dict[str, Any]]
    model: str
    cache_dir: str | None = None
    device: str
    sample_rate: int
    duration_seconds: float
    media_url: str | None = None


def merge_emotion_sources(
    text_emotions: list[dict[str, Any]],
    speech_emotions: list[dict[str, Any]] | None,
    text_weight: float = 0.55,
    speech_weight: float = 0.45,
) -> list[dict[str, Any]]:
    if not speech_emotions:
        return text_emotions

    merged: dict[str, float] = {}
    display_names: dict[str, str] = {}

    def add_items(items: list[dict[str, Any]], weight: float) -> None:
        for item in items:
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            key = normalize_text(name)
            try:
                value = max(0.0, float(item.get("value", 0.0)))
            except Exception:
                value = 0.0
            merged[key] = merged.get(key, 0.0) + value * weight
            display_names.setdefault(key, name)

    add_items(text_emotions, text_weight)
    add_items(speech_emotions, speech_weight)

    ranked = [
        {"name": display_names[key], "value": value}
        for key, value in sorted(merged.items(), key=lambda pair: pair[1], reverse=True)
    ]
    return normalize_emotion_percentages(ranked[:6], decimals=1)


@app.get("/")
def read_root():
    with open(Path(STATIC_DIR) / "index.html", "r", encoding="utf-8") as file:
        return HTMLResponse(
            content=file.read(),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(SOURCE_IMAGE)


@app.get("/graph_data")
def get_graph_data():
    snapshot = load_graph_snapshot()
    return {"nodes": snapshot["nodes"], "links": snapshot["links"]}


@app.post("/speech_emotion", response_model=SpeechEmotionResponse)
async def speech_emotion_endpoint(audio: UploadFile = File(...)):
    if not audio.filename:
        raise HTTPException(status_code=400, detail="请上传音频文件。")

    suffix = Path(audio.filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".webm"}:
        raise HTTPException(status_code=400, detail="不支持的音频格式，请上传 wav/mp3/m4a/aac/ogg/flac/webm。")

    session_id = str(uuid.uuid4())
    saved_path = Path(UPLOAD_DIR) / f"{session_id}{suffix}"
    try:
        with saved_path.open("wb") as output_file:
            while chunk := await audio.read(1024 * 1024):
                output_file.write(chunk)

        result = await asyncio.to_thread(speech_emotion_recognizer.predict_file, saved_path)
        result["media_url"] = None
        return result
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"语音情感识别失败：{exc}") from exc


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="消息不能为空。")
    input_mode = "voice" if str(req.input_mode).lower() == "voice" else "text"
    speech_emotions_data = normalize_emotion_percentages(req.speech_emotions or [], decimals=1) if req.speech_emotions else []

    try:
        emotions_data = await analyze_text_emotions(user_input)
        top_emotions = [item["name"] for item in emotions_data[:2]] or ["平静"]
    except Exception:
        traceback.print_exc()
        emotions_data = [{"name": "平静", "value": 100.0}]
        top_emotions = ["平静"]

    text_emotions_data = emotions_data
    if input_mode == "voice" and speech_emotions_data:
        emotions_data = merge_emotion_sources(text_emotions_data, speech_emotions_data)
        top_emotions = [item["name"] for item in emotions_data[:2]] or top_emotions

    node_filter.reset()
    query_failed = False
    try:
        raw_knowledge = await asyncio.to_thread(
            rag.query,
            user_input,
            param=QueryParam(mode="hybrid", top_k=3),
        )
        retrieved_knowledge = stringify_knowledge(raw_knowledge)
    except Exception:
        traceback.print_exc()
        query_failed = True
        retrieved_knowledge = "暂无相关图谱资料。"

    highlight_nodes = rank_text_items(split_logged_terms(node_filter.last_nodes), limit=15)
    highlight_edges = parse_edge_hints(node_filter.last_edges, limit=18)

    retrieved_knowledge = stringify_knowledge(retrieved_knowledge)
    highlight_nodes, highlight_edges = derive_graph_highlights(
        highlight_nodes,
        highlight_edges,
        user_input if query_failed else retrieved_knowledge,
        limit_nodes=15,
        limit_edges=18,
    )
    graph_grounding = build_graph_grounding_brief(highlight_nodes, highlight_edges, retrieved_knowledge)
    safe_knowledge = retrieved_knowledge[:800]
    if len(retrieved_knowledge) > 800:
        safe_knowledge += "\n...\n(资料已截断)"

    system_prompt = (
        "你是一位专业、温和、稳重的心理支持助手。"
        "请严格依据参考资料和图谱命中信息回答，不要脱离资料泛泛而谈。"
        "如果资料中出现了诱发因素、核心情绪或调节策略，回答里必须自然融入至少1到2个具体点。"
        "如果资料不足，请明确说明资料有限，不要编造不存在的知识。"
        "先共情，再概括问题，再给出1到2条具体、可执行的小建议。"
        "避免网络论坛口吻，保持简洁自然。"
        "只输出纯文本，不要使用 Markdown、加粗、标题、编号、项目符号、星号或代码标记。"
        "全文最多4句话，最多2段。"
        "不要写“根据资料中提到”“图谱显示”等生硬引导语，把信息自然融入回答。"
    )
    current_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"用户当前情绪：{', '.join(top_emotions)}\n"
                f"输入方式：{'语音输入' if input_mode == 'voice' else '文本输入'}\n"
                f"文本情绪识别：{json.dumps(text_emotions_data[:3], ensure_ascii=False)}\n"
                f"语音情绪识别：{json.dumps(speech_emotions_data[:3], ensure_ascii=False) if speech_emotions_data else '未提供'}\n"
                f"图谱命中摘要：\n{graph_grounding}\n\n"
                f"参考资料：\n{safe_knowledge}\n\n"
                f"用户原话：{user_input}\n\n"
                "请先回应用户感受，再结合图谱和资料指出与用户最相关的1到2个因素，"
                "并给出1到2条清晰、具体、可执行的建议。"
                "回答里至少显式提到一个命中的节点名、因素名或策略名；如果没有可靠命中，请明确说明资料有限。"
                "请直接输出最终回答正文：纯文本、无 Markdown、无加粗、无编号、无项目符号、无小标题。"
                "优先写成2到4句自然的话，不要堆成长段落。"
            ),
        },
    ]

    try:
        response_text = await call_siliconflow_api(current_messages, max_tokens=300)
    except Exception as exc:
        traceback.print_exc()
        response_text = f"抱歉，回答生成时出现了问题：{exc}"

    response_text = normalize_generated_reply(response_text)
    speech_text = await build_speech_script(
        response_text,
        user_input=user_input,
        top_emotions=top_emotions,
        graph_grounding=graph_grounding,
        max_chars=55,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    session_id = str(uuid.uuid4())
    audio_path = os.path.join(MEDIA_DIR, f"{session_id}.mp3")
    video_path = os.path.join(MEDIA_DIR, f"{session_id}.mp4")
    media_url = None
    audio_url = None
    media_type = None
    media_status = "文本模式"

    try:
        await generate_voice(speech_text, audio_path)
        media_url = f"/static/media/{session_id}.mp3"
        audio_url = media_url
        media_type = "audio"
        media_status = "语音模式"
    except Exception:
        traceback.print_exc()

    if media_type == "audio":
        generated_video_path = await asyncio.to_thread(generate_digital_human_video, audio_path, video_path)
        if generated_video_path and os.path.exists(generated_video_path):
            media_url = f"/static/media/{Path(generated_video_path).name}"
            media_type = "video"
            media_status = "数字人口型驱动"
        else:
            media_status = "未检测到 EchoMimicV3 / Wav2Lip / Ditto / LivePortrait 视频，已退回语音模式"

    return {
        "reply": response_text,
        "speech_text": speech_text,
        "media_url": media_url,
        "audio_url": audio_url,
        "media_type": media_type,
        "media_status": media_status,
        "emotions": emotions_data,
        "text_emotions": text_emotions_data,
        "speech_emotions": speech_emotions_data,
        "emotion_mode": "multimodal" if input_mode == "voice" and speech_emotions_data else "text",
        "speech_emotion_model": req.speech_emotion_model,
        "highlight_nodes": highlight_nodes,
        "highlight_edges": highlight_edges,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
