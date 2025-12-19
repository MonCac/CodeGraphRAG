import json
import os
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm
from loguru import logger

from codebase_rag.enre.loader import ENRELoader
from codebase_rag.prompts import build_fix_system_prompt, build_generate_file_repair_suggestions_prompt, \
    DIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT, build_generate_file_repair_code_prompt, \
    INDIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT, build_indirect_dependency_change_prompt, \
    build_generate_file_repair_code_prompt_1, build_indirect_dependency_change_prompt_1, build_fix_system_prompt_1
from codebase_rag.services.llm import create_repair_code_model
from codebase_rag.tools.analyze_antipattern_relevance_files import run_with_retry, run_with_retry_code


def generate_direct_file_repair_suggestions(antipattern_json_path, classify_result, antipattern_type,
                                            repaired_description_json_path):
    """
    Step 1: 为直接相关文件生成修复建议（自然语言描述）
    """

    llm_json = ""
    if repaired_description_json_path:
        fix_system_prommpt = build_fix_system_prompt(antipattern_type, repaired_description_json_path)
    else:
        fix_system_prommpt = build_fix_system_prompt_1(antipattern_type)
    repair_code_model = create_repair_code_model(fix_system_prommpt)
    print(f"fix_system_prommpt: {fix_system_prommpt}")
    with open(antipattern_json_path, "r", encoding="utf-8") as f:
        antipattern_json = json.load(f)

    prompt = build_generate_file_repair_suggestions_prompt(classify_result, antipattern_json, antipattern_type)
    print(f"original_prompt: {prompt}")
    print("original_prompt Out over")

    direct_related = classify_result.get("direct_related", [])

    MAX_OUTER_RETRIES = 2
    for attempt in range(1, MAX_OUTER_RETRIES + 1):
        try:
            llm_json = run_with_retry(repair_code_model, prompt, max_retries=2)
            if is_valid_repair_json(llm_json, direct_related):
                break
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
        # 构造下一轮 prompt（显式纠错）
        prompt = f"""
        你刚才的输出 JSON 不符合要求。

        【原始任务】
        {prompt}

        【你的上一次输出】
        {json.dumps(llm_json, ensure_ascii=False, indent=2)}

        【问题】
        - 必须包含 summary
        - files 不能为空
        - 每个 file 必须包含 file_path 和 repair_description

        请重新生成，并且只返回 JSON。
        """

    filter_files_by_direct_related(llm_json, direct_related)
    return llm_json


def is_valid_repair_json(obj: dict, direct_related) -> bool:
    # ---- 1. 基本结构校验 ----
    if not isinstance(obj, dict):
        return False

    if "summary" not in obj or "files" not in obj:
        return False

    if not isinstance(obj["files"], list) or len(obj["files"]) == 0:
        return False

    file_paths = []

    for f in obj["files"]:
        if not isinstance(f, dict):
            return False
        if not f.get("file_path"):
            return False
        if not f.get("repair_description"):
            return False
        file_paths.append(f["file_path"])

    # ---- 2. direct_related 覆盖校验 ----
    if not isinstance(direct_related, list) or not direct_related:
        # 没有 direct_related，不做覆盖校验
        return True

    llm_path_set = set(file_paths)
    direct_set = set(direct_related)

    # 必须包含 direct_related 中的所有文件
    if not direct_set.issubset(llm_path_set):
        return False

    return True


def filter_files_by_direct_related(llm_json: dict, direct_related) -> dict:
    if not isinstance(llm_json, dict):
        return llm_json

    files = llm_json.get("files")
    if not isinstance(files, list):
        return llm_json

    # 用 set 加速查找
    direct_set = set(direct_related)

    filtered_files = []
    for f in files:
        if not isinstance(f, dict):
            continue

        file_path = f.get("file_path")
        if file_path in direct_set:
            filtered_files.append(f)

    llm_json["files"] = filtered_files
    return llm_json


def generate_direct_file_code_repair(
        target_repo_path,
        direct_suggestions,
        output_dir,
        other_files,
        antipattern_type,
        result_folder_name,
        max_attempts: int = 3,
):
    """
    Step 2: 根据修复建议生成具体代码修复（支持 GAP 失败重试）
    - 最多重试 max_attempts 次
    - 第 max_attempts 次仍失败 => 最终标记为失败
    """

    antipattern_type = antipattern_type.upper()
    after_code_folder_name = result_folder_name
    after_code_path = os.path.join(output_dir, after_code_folder_name)
    other_content = analyze_other_info(other_files, output_dir)

    print(f"other_content: {other_content}")
    attempt = 0
    GAP_test = 0
    success = False
    last_gap_json_path = None
    last_gap_count = None

    while attempt < max_attempts:
        attempt += 1
        logger.info(f"===== Attempt {attempt}/{max_attempts} =====")

        # 1️⃣ 非第一次先删除旧结果
        if attempt > 1:
            remove_dir_if_exists(after_code_path)

        # ================= 生成修复代码 =================
        client = create_repair_code_model(DIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT)
        summary = direct_suggestions.get("summary")
        files = direct_suggestions.get("files", [])

        for file_info in tqdm(files, desc="Generating direct file code repair"):
            user_input = build_generate_file_repair_code_prompt(target_repo_path, summary, file_info)
            # 加上无关文件的
            user_input = build_generate_file_repair_code_prompt_1(target_repo_path, summary, file_info, other_content)

            try:
                new_code = run_with_retry_code(client, user_input)
                new_code = strip_markdown_code_block(new_code)
            except Exception as e:
                logger.exception("LLM 调用失败")
                new_code = "llm return wrong"

            file_path = file_info.get("file_path")
            if not file_path:
                logger.warning("file_path is empty, skip")
                continue

            tmp_file_path = os.path.join(after_code_path, file_path)
            os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

            with open(tmp_file_path, "w", encoding="utf-8") as f:
                f.write(new_code)

        # ================= ENRE =================
        enre_path = Path("codebase_rag/enre/lib/enre_java.jar").resolve()
        enre_command = (
            f"java -Xmx20G -jar {enre_path} "
            f"java {after_code_path} {after_code_folder_name}"
        )

        cwd = os.getcwd()
        enre_out_path = os.path.join(cwd, f"{after_code_folder_name}-enre-out")

        try:
            subprocess.run(enre_command, shell=True, check=True)
            move_folder(enre_out_path, after_code_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"ENRE 执行失败: {e}")
            break

        # ================= GAP =================
        gap_path = Path("codebase_rag/enre/lib/GAP-1.0.jar").resolve()

        enre_json_files = list(Path(after_code_path).rglob("*-out.json"))
        print(f"after_code_path: {after_code_path}")
        print(f"数目：{len(enre_json_files)}")
        print(f"enre_json_files：{enre_json_files}")
        if len(enre_json_files) != 1:
            raise RuntimeError(f"ENRE JSON 数量异常: {enre_json_files}")

        enre_json_path = enre_json_files[0].resolve()

        gap_command = (
            f"java -jar {gap_path} detect "
            f"-n {after_code_folder_name} "
            f"-d {enre_json_path}"
        )

        gap_out_path = os.path.join(cwd, f"{after_code_folder_name}-gap-out")

        try:
            subprocess.run(gap_command, shell=True, check=True)
            move_folder(gap_out_path, after_code_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"GAP 执行失败: {e}")
            break

        # ================= 读取 GAP 结果 =================
        gap_json_files = list(
            Path(after_code_path).rglob(f"*{antipattern_type}.json")
        )
        if len(gap_json_files) != 1:
            raise RuntimeError(f"GAP JSON 数量异常: {gap_json_files}")

        gap_json_path = gap_json_files[0].resolve()
        last_gap_json_path = gap_json_path

        last_gap_count = read_gap_count(gap_json_path)
        logger.info(f"GAP count = {last_gap_count}")

        # ================= 判断是否成功 =================
        if last_gap_count == 0:
            success = True
            logger.info("✅ GAP 检测通过，修复成功")
            break
        else:
            logger.warning("❌ GAP 仍检测到反模式，将重试")

    # ================= 最终记录 =================
    result = {
        "antipattern_type": antipattern_type,
        "max_attempts": max_attempts,
        "attempts": attempt,
        "success": success,
        "final_gap_count": last_gap_count,
        "final_gap_json_path": str(last_gap_json_path)
        if last_gap_json_path
        else None,
    }

    result_path = os.path.join(output_dir, result_folder_name, "repair_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if not success:
        logger.error(
            f"最终失败：antipattern={antipattern_type}, "
            f"attempts={attempt}, count={last_gap_count}"
        )

    return result


def remove_dir_if_exists(path: str):
    if not os.path.isdir(path):
        return

    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)


def read_gap_count(json_path) -> int:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("count", 0))


def move_folder(before_folder, after_folder):
    """
    将整个 before_folder 文件夹移动到 after_folder 下。
    如果 after_folder 中已存在 before_folder 的 basename 文件夹，则先删除后移动。
    """
    try:
        if not os.path.exists(before_folder):
            raise FileNotFoundError(f"源文件夹不存在: {before_folder}")

        # 目标完整路径是 after_folder/basename(before_folder)
        target_path = os.path.join(after_folder, os.path.basename(before_folder))

        # 如果目标存在，先删除
        if os.path.exists(target_path):
            shutil.rmtree(target_path)

        # 创建目标父目录
        os.makedirs(after_folder, exist_ok=True)

        # 移动整个文件夹
        shutil.move(before_folder, target_path)
        print(f"✅ 成功移动：{before_folder} → {target_path}")

        return target_path

    except Exception as e:
        print(f"❌ 移动失败: {e}")


def strip_markdown_code_block(text: str) -> str:
    """
    去掉 LLM 返回的 markdown 代码块包裹：
    ```java
    ...
    ```
    """
    if not text:
        return text

    text = text.strip()

    # 匹配 ```lang\n ... \n```
    pattern = r"^```[a-zA-Z0-9_+-]*\n([\s\S]*?)\n```$"
    match = re.match(pattern, text)

    if match:
        return match.group(1)

    return text


def analyze_indirect_files_for_changes(target_repo_path, classify_result, direct_suggestions, output_dir, other_files, antipattern_type, result_folder_name):
    """
    Step 3: 判断间接相关文件是否需要修改；如需修改，生成具体代码
    """
    updated_files = []
    client = create_repair_code_model(INDIRECT_FILE_CODE_REPAIR_SYSTEM_PROMPT)
    indirect_files = classify_result.get("indirect_related", [])
    summary = direct_suggestions.get("summary")
    modification_scope = direct_suggestions.get("modification_scope", [])
    after_code_path = os.path.join(output_dir, result_folder_name)

    if "subclass" not in modification_scope:
        return {
            "updated_files": []
        }

    other_content = analyze_other_info(other_files, output_dir)
    i = 0
    for file_path in tqdm(indirect_files, desc="Analyzing indirect files for changes"):
        i += 1
        if i > 4:
            break
        abs_path = os.path.join(target_repo_path, file_path)
        with open(abs_path, 'r', encoding='utf-8') as f:
            indirect_code = f.read()

        # 构造 LLM prompt
        user_input = build_indirect_dependency_change_prompt(summary, indirect_code, antipattern_type)

        user_input = build_indirect_dependency_change_prompt_1(summary, indirect_code, antipattern_type, other_content)

        try:
            llm_output = run_with_retry(client, user_input)

            # llm_output 结构示例：
            # {
            #     "should_update": true/false,
            #     "reason": "...",
            #     "patched_code": "..."  # 如果 should_update = true
            # }

        except Exception as e:
            continue
        should_update = llm_output.get("should_update", False)

        if should_update:
            updated_files.append(file_path)
            new_code = strip_markdown_code_block(llm_output.get("file_content"))
            tmp_file_path = os.path.join(after_code_path, file_path)
            os.makedirs(os.path.dirname(tmp_file_path), exist_ok=True)

            with open(tmp_file_path, "w", encoding="utf-8") as f:
                f.write(new_code)

    return updated_files


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


def load_selected_files(jsonl_path, output_dir):
    selected_files = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "file_path" in obj:
                file_path = os.path.join(obj["repo_path"], obj["file_path"])
                selected_files.append(file_path)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    temp_dir = os.path.join(project_root, output_dir)
    temp_dir = tempfile.mkdtemp(prefix="enre_selected_repo_", dir=temp_dir)

    for full_path in selected_files:
        filename = os.path.basename(full_path)
        dst = os.path.join(temp_dir, filename)
        shutil.copy2(full_path, dst)

    print(f"Temporary flat repo created at: {temp_dir}")

    print("=== Running ENRELoader Test ===")
    try:
        loader = ENRELoader(temp_dir, output_dir)
        loader.run_enre_analysis()
    finally:
        shutil.rmtree(temp_dir)
        print(f"Temporary repo deleted: {temp_dir}")
    return loader.json_path


def analyze_other_info(jsonl_path, output_dir):
    with open(load_selected_files(jsonl_path, output_dir), "r", encoding="utf-8") as f:
        entities_data = json.load(f)

    file_priority = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            file_priority.append(json.loads(line))

    other_content = other_info(
        entities_data["variables"],
        file_priority,
        sample_ratio=0.5,
        max_class=2,
        max_method=5
    )

    return other_content
