import json
import os
import random
import shutil
import tempfile

from tqdm import tqdm
from loguru import logger

from codebase_rag.enre.loader import ENRELoader
from codebase_rag.prompts import build_fix_system_prompt, build_generate_file_repair_suggestions_prompt, \
    DIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT, build_generate_file_repair_code_prompt, \
    INDIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT, build_indirect_dependency_change_prompt, \
    build_generate_file_repair_code_prompt_1, build_indirect_dependency_change_prompt_1
from codebase_rag.services.llm import create_repair_code_model
from codebase_rag.tools.analyze_antipattern_relevance_files import run_with_retry, run_with_retry_code

other_files = "D:\\Disaster\\Codefield\\Code_Python\\CodeGraphRAG\\codebase_rag\\services\\embedding\\output2.jsonl"


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

    other_content = analyze_other_info(other_files)

    files = direct_suggestions.get("files", [])
    for file_info in tqdm(files, desc="Generating direct file code repair"):
        user_input = build_generate_file_repair_code_prompt(target_repo_path, summary, file_info)
        user_input_1 = build_generate_file_repair_code_prompt_1(target_repo_path, summary, file_info, other_content)
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

    other_content = analyze_other_info(other_files)

    for file_path in tqdm(indirect_files, desc="Analyzing indirect files for changes"):
        abs_path = os.path.join(target_repo_path, file_path)
        with open(abs_path, 'r', encoding='utf-8') as f:
            indirect_code = f.read()

        # 构造 LLM prompt
        user_input = build_indirect_dependency_change_prompt(summary, indirect_code)

        user_input_1 = build_indirect_dependency_change_prompt_1(summary, indirect_code, other_content)

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


def other_info(json_data, file_priority_list, sample_ratio=1.0, max_class=None, max_method=None):
    file_score_map = {
        item["file_path"].replace("\\", "/"): item["score"]
        for item in file_priority_list
    }

    skip_keys = {"id", "parentId", "additionalBin", "external", "location"}

    class_items = []
    method_items = []

    for item in json_data:
        if not isinstance(item, dict):
            continue
        category = item.get("category")
        if category not in ("Class", "Method"):
            continue

        pkg = item["qualifiedName"].rsplit(".", 1)[0].replace(".", "/")
        file_name = item.get("File")
        if not file_name:
            continue

        matched_path = None
        for full_path in file_score_map.keys():
            if pkg in full_path and full_path.endswith("/" + file_name):
                matched_path = full_path
                break

        weight = file_score_map.get(matched_path, 0.0001)

        clean_item = {
            k: v for k, v in item.items()
            if k not in skip_keys
        }
        clean_item["_weight"] = weight

        if category == "Class":
            class_items.append(clean_item)
        else:
            method_items.append(clean_item)

    def weighted_sample(entity_list):
        if not entity_list:
            return []

        entity_list.sort(key=lambda x: x["_weight"], reverse=True)

        if sample_ratio >= 1.0:
            for x in entity_list:
                x.pop("_weight", None)
            return entity_list

        weights = [x["_weight"] for x in entity_list]
        keep_count = max(1, int(len(entity_list) * sample_ratio))

        sampled = random.choices(
            population=entity_list,
            weights=weights,
            k=keep_count
        )

        for x in sampled:
            x.pop("_weight", None)

        return sampled

    sampled_class = weighted_sample(class_items)
    sampled_method = weighted_sample(method_items)

    if max_class is not None and sampled_class:
        # 优先保留权重高的
        sampled_class = sorted(
            sampled_class,
            key=lambda x: x.get("_weight", 0),
            reverse=True
        )[:max_class]

    if max_method is not None and sampled_method:
        sampled_method = sorted(
            sampled_method,
            key=lambda x: x.get("_weight", 0),
            reverse=True
        )[:max_method]

    return sampled_class + sampled_method


def load_selected_files(jsonl_path):
    selected_files = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "file_path" in obj:
                file_path = os.path.join(obj["repo_path"], obj["file_path"])
                selected_files.append(file_path)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    temp_dir = os.path.join(project_root, "tmp")
    temp_dir = tempfile.mkdtemp(prefix="enre_selected_repo_", dir=temp_dir)

    for full_path in selected_files:
        filename = os.path.basename(full_path)
        dst = os.path.join(temp_dir, filename)
        shutil.copy2(full_path, dst)

    print(f"Temporary flat repo created at: {temp_dir}")

    print("=== Running ENRELoader Test ===")
    try:
        loader = ENRELoader(temp_dir)
        loader.run_enre_analysis()
    finally:
        shutil.rmtree(temp_dir)
        print(f"Temporary repo deleted: {temp_dir}")
    return loader.json_path


def analyze_other_info(jsonl_path):
    with open(load_selected_files(jsonl_path), "r", encoding="utf-8") as f:
        entities_data = json.load(f)

    file_priority = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            file_priority.append(json.loads(line))

    other_content = other_info(
        entities_data["variables"],
        file_priority,
        sample_ratio=0.5,
        max_class=100,
        max_method=200
    )

    return other_content
