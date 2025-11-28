import json
import os
from pathlib import Path

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


def run_with_retry_code(client, user_prompt, max_retries=3):
    attempt = 0
    while attempt <= max_retries:
        try:
            result = client.run_sync(user_prompt)
            print(result)
            # LLM 返回的内容（通常是字符串），去除多余空白
            desc = getattr(result, "output", str(result)).strip()
            # 直接返回字符串，不解析JSON
            return desc
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise e
            else:
                logger.warning(f"LLM 调用失败（第 {attempt} 次重试），错误：{e}")


def analyze_files_with_llm(file_result_json_path, antipattern_to_update, repo_to_update):
    """
    与 LLM 交互，判断 file_result.json 中每个文件是否与反模式修复相关。

    Args:
        file_result_json_path (str): .tmp/file_result.json 路径
        antipattern_to_update (str | Path): .env 中定义的反模式目录路径
        repo_to_update (str | Path): 项目根目录绝对路径，用于拼接文件绝对路径

    Returns:
        list: 最终与反模式修复相关的文件信息列表
    """
    # 1. 读取 file_result.json，取 indirect_related 文件列表
    with open(file_result_json_path, 'r', encoding='utf-8') as f:
        file_result_data = json.load(f)
    indirect_related_files = file_result_data.get("indirect_related", [])

    # 2. 找到 antipattern.json，位于 antipattern_to_update 的父目录下
    antipattern_json_path = None
    antipattern_parent_dir = Path(antipattern_to_update).parent
    for filename in os.listdir(antipattern_parent_dir):
        if filename.endswith("antipattern.json"):
            antipattern_json_path = antipattern_parent_dir / filename
            break
    if antipattern_json_path is None:
        raise FileNotFoundError(f"antipattern.json not found in {antipattern_parent_dir}")

    # 3. 读取 antipattern.json 内容
    with open(antipattern_json_path, 'r', encoding='utf-8') as f:
        antipattern_data = json.load(f)

    relevant_files = []

    # 4. 遍历 indirect_related 文件，构造绝对路径，读取内容，调用 LLM 判断
    for rel_path in tqdm(indirect_related_files, desc="分析文件中", unit="file"):
        abs_file_path = Path(repo_to_update) / rel_path

        # 读取文件内容（你可以根据需要改成只读部分）
        try:
            with open(abs_file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except Exception as e:
            # 文件读不了时，记录异常信息，继续下一个文件
            relevant_files.append({
                "file": str(abs_file_path),
                "involved_in_antipattern_repair": False,
                "reason": f"读取文件失败: {e}"
            })
            continue

        client = create_relevance_model()

        # 构造传给LLM的输入，假设 build_antipattern_relevance_user_input_func 需要文件内容和反模式数据
        user_prompt = build_antipattern_relevance_user_input_func(file_path=str(abs_file_path),
                                                                  file_content=file_content,
                                                                  antipattern_data=antipattern_data)

        try:
            llm_json = run_with_retry(client, user_prompt, max_retries=2)
            llm_json["file"] = str(abs_file_path)
        except Exception as e:
            llm_json = {
                "file": str(abs_file_path),
                "involved_in_antipattern_repair": False,
                "reason": f"LLM调用失败: {e}"
            }

        relevant_files.append(llm_json)

    return relevant_files
