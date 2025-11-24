import json
import os
from tqdm import tqdm
from loguru import logger

from codebase_rag.prompts import build_fix_system_prompt, build_generate_file_repair_suggestions_prompt, \
    DIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT, build_generate_file_repair_code_prompt, \
    INDIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT, build_indirect_dependency_change_prompt
from codebase_rag.services.llm import create_repair_code_model
from codebase_rag.tools.analyze_antipattern_relevance_files import run_with_retry, run_with_retry_code


def generate_direct_file_repair_suggestions(antipattern_json_path, classify_result, antipattern_type):
    """
    Step 1: 为直接相关文件生成修复建议（自然语言描述）
    """

    repair_code_model = create_repair_code_model(build_fix_system_prompt(antipattern_type))

    with open(antipattern_json_path, "r", encoding="utf-8") as f:
        antipattern_json = json.load(f)

    user_input = build_generate_file_repair_suggestions_prompt(classify_result, antipattern_json)

    try:
        llm_json = run_with_retry(repair_code_model, user_input, max_retries=2)
    except Exception as e:
        # 返回符合预期JSON格式，但内容表示失败
        return {
            "summary": "LLM调用失败，无法生成修复建议。",
            "files": [
                {
                    "file_path": None,
                    "repair_description": f"LLM调用失败: {e}"
                }
            ]
        }

    return llm_json


def generate_direct_file_code_repair(target_repo_path, direct_suggestions):
    """
    Step 2: 根据修复建议生成具体代码修复
    """
    client = create_repair_code_model(DIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT)
    results = []
    summary = direct_suggestions.get("summary")

    files = direct_suggestions.get("files", [])
    for file_info in tqdm(files, desc="Generating direct file code repair"):
        user_input = build_generate_file_repair_code_prompt(target_repo_path, summary, file_info)
        try:
            new_code = run_with_retry_code(client, user_input)
        except Exception as e:
            new_code = "llm return wrong"

        # 生成内容的存储
        file_path = file_info.get("file_path")
        if not file_path:
            # 处理异常情况，比如跳过
            logger.warning("file_path is None or empty, skipping this file_info")
            continue
        tmp_file_path = os.path.join("tmp", file_path)
        os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

        with open(tmp_file_path, "w", encoding="utf-8") as f:
            f.write(new_code)


def analyze_indirect_files_for_changes(target_repo_path, classify_result, direct_suggestions):
    """
    Step 3: 判断间接相关文件是否需要修改；如需修改，生成具体代码
    """
    results = []
    client = create_repair_code_model(INDIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT)
    indirect_files = classify_result.get("indirect_related", [])
    summary = direct_suggestions.get("summary")
    for file_path in tqdm(indirect_files, desc="Analyzing indirect files for changes"):
        abs_path = os.path.join(target_repo_path, file_path)
        with open(abs_path, 'r', encoding='utf-8') as f:
            indirect_code = f.read()

        # 构造 LLM prompt
        user_input = build_indirect_dependency_change_prompt(summary, indirect_code)

        try:
            llm_output = run_with_retry(client, user_input)

            # llm_output 结构示例：
            # {
            #     "should_update": true/false,
            #     "reason": "...",
            #     "patched_code": "..."  # 如果 should_update = true
            # }

        except Exception as e:
            llm_output = {
                "should_update": False,
                "file_content": f"LLM failed: {e}",
            }

        results.append(llm_output)

    return results
