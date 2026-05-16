import os
import json
import argparse
from pathlib import Path

from app.knowledge_base import rag
from lightrag.llm.openai import openai_complete_if_cache


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "knowledge" / "data_pro.json"

# 在构建新数据时，动态把内部抽取模型临时更换为最强大的 72B-Instruct
async def extraction_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    from app.knowledge_base import SILICONFLOW_API_KEY
    return await openai_complete_if_cache(
        "Qwen/Qwen2.5-72B-Instruct",  # 构建图谱时使用 72B 保证最高质量
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=SILICONFLOW_API_KEY,
        base_url="https://api.siliconflow.cn/v1",
        **kwargs
    )

def main():
    rag.llm_model_func = extraction_llm_model_func  # 劫持注入 72B 抽取模型
    
    parser = argparse.ArgumentParser(description="增量或全量构建 LightRAG 知识图谱 (仅抽取)")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA_PATH),
                        help="增量或全量数据的路径（JSON文件）")
    args = parser.parse_args()

    data_path = Path(args.data).expanduser()
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()
    if not os.path.exists(data_path):
        print(f"❌ 未找到对应的数据文件: {data_path}。")
        print("请检查路径，或将想要抽取的新文本放入此位置。")
        return

    print(f"========== 正在从 {data_path} 增量/全量构建 LightRAG 知识图谱 ==========")
    print("这可能需要一些时间（正在调用配置的 LLM 并抽取）...")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # 根据 JSON 格式处理为文本列表
        if isinstance(data, list):
            texts = [json.dumps(item, ensure_ascii=False) for item in data]
        else:
            texts = [json.dumps(data, ensure_ascii=False)]
        
        # 将最新的段落推入 LightRAG
        rag.insert(texts)
    
    print("\n✅ 该批文本数据已成功提取，知识图谱已更新保存至本地工作区！")
    print("现在您可以运行 python web_app.py 来体验基于最新知识图谱的问答。")

if __name__ == "__main__":
    main()
