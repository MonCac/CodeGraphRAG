import json
import os
from loguru import logger
from tqdm import tqdm

from codebase_rag.prompts import build_antipattern_relevance_user_input_func
from codebase_rag.services.llm import create_relevance_model


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


def analyze_files_with_llm(file_result_json_path, antipattern_dir_path):
    """
    与 LLM 交互，判断 file_result.json 中每个文件是否与反模式修复相关。

    Args:
        file_result_json_path (str): .tmp/file_result.json 路径
        antipattern_dir_path (str): .env 中定义的反模式目录路径

    Returns:
        list: 最终与反模式修复相关的文件信息列表
    """
    # 1. 读取 file_result.json
    with open(file_result_json_path, 'r', encoding='utf-8') as f:
        file_result_data = json.load(f)

    candidate_files = file_result_data.get("metadatas", [])

    # 2. 读取 antipattern 目录下唯一的 *antipattern.json 文件
    antipattern_data = None
    for filename in os.listdir(antipattern_dir_path):
        if filename.endswith("antipattern.json"):
            antipattern_json_path = os.path.join(antipattern_dir_path, filename)
            with open(antipattern_json_path, 'r', encoding='utf-8') as f:
                antipattern_data = json.load(f)
            break

    # 3. 遍历每个候选文件，调用 LLM 判断是否相关
    relevant_files = []
    for candidate_file in tqdm(candidate_files, desc="分析文件中", unit="file"):
        client = create_relevance_model()
        user_prompt = build_antipattern_relevance_user_input_func(candidate_file, antipattern_data)

        try:
            llm_json = run_with_retry(client, user_prompt, max_retries=2)

            # 构造 file 字段
            file_path = candidate_file.get("properties", {}).get("additionalBin", {}).get("binPath", "")
            file_name = candidate_file.get("properties", {}).get("name", "")
            llm_json["file"] = f"{file_path}/{file_name}"

        except Exception as e:
            logger.warning(f"LLM进行 antipattern relevance 判断失败：{e}")
            file_path = candidate_file.get("properties", {}).get("additionalBin", {}).get("binPath", "")
            file_name = candidate_file.get("properties", {}).get("name", "")
            llm_json = {
                "involved_in_antipattern_repair": False,
                "reason": f"LLM调用失败: {e}",
                "file": f"{file_path}/{file_name}"
            }

        relevant_files.append(llm_json)

    return relevant_files
