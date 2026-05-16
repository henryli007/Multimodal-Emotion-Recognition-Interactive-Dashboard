import json
import os
import warnings
from pathlib import Path
from lightrag import LightRAG
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.prompt import PROMPTS

warnings.filterwarnings('ignore')

# ==========================================
# LightRAG 与 SiliconFlow API 核心配置层
# ==========================================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKING_DIR = PROJECT_ROOT / "workspace" / "lightrag"
WORKING_DIR.mkdir(parents=True, exist_ok=True)

async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    # 强制在提示词最后加上语言约束，避免国外模型串台到英文提取
    if system_prompt:
        system_prompt += "\n重要：无论输入是什么语言，你的所有输出（包括关键词提取和总结）必须使用简体中文！"
    
    return await openai_complete_if_cache(
        "Qwen/Qwen2.5-7B-Instruct",  # 检索阶段改为免费模型
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        **kwargs
    )

async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed.func(
        texts,
        model="BAAI/bge-m3",
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        embedding_dim=1024,
    )

user_rules = """
请绝对遵循以下规则：
1. 【核心关注】：你必须将注意力 100% 集中在提取文本中的“情绪状态”、“心理症状”、“诱发原因”以及“干预/疏导举措”上。
2. 【过滤噪音】：忽略文本中无关的日常琐事、时间、地点、普通人称代词。
3. 【实体类型限制】：你提取的实体类型（entity_type）必须严格属于以下类别之一：
   - 情绪/症状 (Emotion/Symptom)
   - 诱发因素 (Trigger)
   - 疏导举措 (Intervention_Method)
   - 治疗流派 (Therapy_Approach)
4. 【关系抽取限制】：实体之间的关系（relationship）应重点描述因果与治疗关系，如：“引发”、“缓解”、“适用于”、“包含”。
5. 【语言要求】：请确保提取的所有实体名称、实体描述、关系名以及关系描述等输出，必须完全使用中文。
"""
PROMPTS["entity_extraction_system_prompt"] = PROMPTS["entity_extraction_system_prompt"].replace(
    "---Instructions---",
    f"---Instructions---\n{user_rules}\n"
)

# 【核心功能】：对外导出的 RAG 全局实例
rag = LightRAG(
    working_dir=str(WORKING_DIR),
    llm_model_func=llm_model_func,
    llm_model_max_async=4,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=embedding_func
    ),
    addon_params={
        "entity_types": ["情绪/症状 (Emotion/Symptom)", "诱发因素 (Trigger)", "疏导举措 (Intervention_Method)", "治疗流派 (Therapy_Approach)"]
    }
)
