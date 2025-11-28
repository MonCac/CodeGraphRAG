import json

from loguru import logger
import os

from codebase_rag.main import save_antipattern_relevance_result, _update_model_settings
from codebase_rag.tools.antipattern_repair_llm import generate_direct_file_repair_suggestions, \
    generate_direct_file_code_repair, analyze_indirect_files_for_changes

_update_model_settings("qwen3:0.6b", "qwen3:0.6b", None)


def run_repair_pipeline(antipattern_json_path, classify_result, antipattern_type, target_repo_path):
    logger.info("[bold green]Step 1: Generating repair suggestions for directly related files[/bold green]")
    direct_file_repair_suggestions = generate_direct_file_repair_suggestions(
        antipattern_json_path, classify_result, antipattern_type)
    # direct_file_repair_suggestions_json_path = save_antipattern_relevance_result(
    #     direct_file_repair_suggestions,
    #     os.path.join("tmp", "direct_file_repair_suggestions.json")
    # )
    # logger.info(f"Direct file repair suggestions saved to: {direct_file_repair_suggestions_json_path}")

    logger.info("[bold green]Step 2: Generating concrete code repairs based on suggestions[/bold green]")
    generate_direct_file_code_repair(target_repo_path, direct_file_repair_suggestions)
    logger.info("Direct file code repairs generated successfully.")

    logger.info("[bold green]Step 3: Analyzing indirect files for necessary changes and generating repairs[/bold green]")
    indirect_files_for_changes_results = analyze_indirect_files_for_changes(
        target_repo_path, classify_result, direct_file_repair_suggestions)
    # indirect_files_for_changes_results_json_path = save_antipattern_relevance_result(
    #     indirect_files_for_changes_results,
    #     os.path.join("tmp", "indirect_files_for_changes_results.json")
    # )
    # logger.info(f"Indirect file repair results saved to: {indirect_files_for_changes_results_json_path}")

    # return {
    #     "direct_suggestions_path": direct_file_repair_suggestions_json_path,
    #     "indirect_results_path": indirect_files_for_changes_results_json_path,
    # }

if __name__ == "__main__":
    antipattern_json_path = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/test-project/AWD/86/kafka_86_awd_antipattern.json"
    classify_path = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/tmp/awd-classify_result.json"
    antipattern_type = "awd"
    target_repo_path = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/test-project/AWD/kafka"

    with open(classify_path, "r", encoding="utf-8") as f:
        classify_result = json.load(f)

    run_repair_pipeline(antipattern_json_path, classify_result, antipattern_type, target_repo_path)