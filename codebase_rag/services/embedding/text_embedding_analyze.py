import json
import os

from loguru import logger
from tqdm import tqdm

from codebase_rag.prompts import build_code2text_user_input_func
from codebase_rag.services.llm import create_code2text_model

# output_file = "D:\\Disaster\\Codefield\\Code_Python\\CodeGraphRAG\\codebase_rag\\services\\embedding\\output_code2text.jsonl"

def run_with_retry(client, user_prompt, max_retries=3):
    attempt = 0
    while attempt <= max_retries:
        try:
            result = client.run_sync(user_prompt)
            desc = getattr(result, "output", str(result)).strip()  # LLM 原始返回
            llm_json = json.loads(desc)  # 解析 JSON
            return llm_json
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise e
            else:
                logger.warning(f"LLM 调用失败（第 {attempt} 次重试），错误：{e}")


def analyze_files_with_llm(file_nodes, output_file):
    results = []

    for node in tqdm(file_nodes, desc="代码摘要生成中", unit="file"):
        client = create_code2text_model()
        user_prompt = build_code2text_user_input_func(node)

        try:
            llm_json = run_with_retry(client, user_prompt, max_retries=2)

            # 补充 metadata（若 LLM 输出中已存在则不覆盖）
            llm_json["repo_path"] = node["repo_path"]
            llm_json["file_path"] = node["file_path"]

        except Exception as e:
            logger.warning(f"LLM code2text 失败: {node['file_path']} 错误={e}")
            llm_json = {
                "repo_path": node["repo_path"],
                "file_path": node["file_path"],
                "error": str(e),
            }
        print(llm_json)
        results.append(llm_json)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return results
