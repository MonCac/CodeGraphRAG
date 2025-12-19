import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from loguru import logger

from codebase_rag.main import start


def run_batch(antipattern_type_limit, ablation_type):
    """
    repos = [
        {
            "repo_path": "/path/to/repo1",
            "antipattern_relation_path": "/path/to/anti1.json",
            "antipattern_type": "awd",
        },
        ...
    ]
    """

    # for cfg in repos:
    #     print(f"Processing {cfg['repo_path']}")
    #
    #     start(
    #         repo_path=cfg["repo_path"],
    #         antipattern_relation_path=cfg.get("antipattern_relation_path"),
    #         update_project_graph=True,
    #         update_antipattern_graph=True,
    #         antipattern_type="awd",
    #         semantic_enhance=True,
    #         hybrid_query=False,
    #         clean=False,
    #         output=f"tmp/{cfg['repo_path'].split('/')[-1]}-result.json",
    #         no_confirm=True,
    #     )

    start_default(antipattern_type_limit, ablation_type)


def start_default(
        antipattern_type_limit,
        ablation_type,
        *,
        repo_path: str | None = None,
        antipattern_relation_path: str | None = None,
        update_project_graph: bool = False,
        update_antipattern_graph: bool = True,
        antipattern_type: str = "ch",
        semantic_enhance: bool = True,
        hybrid_query: bool = False,
        clean: bool = True,
        output: str = "tmp/ch-final-result.json",
        orchestrator_model: str | None = None,
        cypher_model: str | None = None,
        embedding_model: str | None = None,
        no_confirm: bool = True,
        result_folder_name: str | None = None,
) -> None:
    pairs = collect_antipattern_repo_pairs("/data/sanglei/反模式修复数据集构建")
    print(f"Total pairs: {len(pairs)}")
    failed_log_path = os.path.join(
        "tmp",
        "failed_cases",
        f"{ablation_type}.jsonl"
    )
    for i, p in enumerate(pairs):
        antipattern_type = p["antipattern_type"].lower()
        project_name = p["project_name"]
        commit_number = p["commit_number"]
        id = p["id"]
        result_folder_name = ablation_type
        output = os.path.join(
            "tmp",
            antipattern_type,
            "apache",
            project_name,
            commit_number,
            str(id),
            "save.json"
        )
        try:
            if antipattern_type == antipattern_type_limit:
                start(
                    repo_path=p["target_repo_path"],
                    antipattern_relation_path=p["antipattern_relation_path"],
                    update_project_graph=update_project_graph,
                    update_antipattern_graph=update_antipattern_graph,
                    antipattern_type=antipattern_type,
                    semantic_enhance=semantic_enhance,
                    hybrid_query=hybrid_query,
                    clean=clean,
                    output=output,
                    orchestrator_model=orchestrator_model,
                    cypher_model=cypher_model,
                    embedding_model=embedding_model,
                    no_confirm=no_confirm,
                    result_folder_name=result_folder_name,
                )
        except Exception as e:
            logger.exception(
                f"[ERROR] Failed case: "
                f"antipattern={antipattern_type}, "
                f"project={project_name}, "
                f"commit={commit_number}, "
                f"id={id}"
            )

            record_failed_case(
                log_path=failed_log_path,
                antipattern_type=antipattern_type,
                project_name=project_name,
                commit_number=commit_number,
                id=id,
                result_folder_name=result_folder_name,
                output=output,
                exception=e,
            )
            continue


def collect_antipattern_repo_pairs(
        base_root: str,
) -> List[Dict[str, str]]:
    """
    Traverse antipattern dataset structure and collect mappings between
    antipattern_relation_path (before) and target_repo_path.

    Returns a list of dicts:
    {
        "antipattern_type": "AWD",
        "project_name": "alluxio",
        "commit_number": 1100,
        "id": 18,
        "antipattern_relation_path": ".../before",
        "target_repo_path": ".../dataset_programs/apache/commit_xxx_snapshot/project"
    }
    """

    base_root = Path(base_root)

    antipattern_root = (
            base_root / "extract_antipatterns_and_repair" / "final"
    )
    dataset_programs_root = (
            base_root / "dataset_programs" / "apache"
    )

    results: List[Dict[str, str]] = []

    # Traverse antipattern types: AWD / CH / MH
    for antipattern_type_dir in antipattern_root.iterdir():
        if not antipattern_type_dir.is_dir():
            continue

        antipattern_type = antipattern_type_dir.name

        apache_dir = antipattern_type_dir / "apache"
        if not apache_dir.exists():
            continue

        # project_name level (e.g., alluxio)
        for project_dir in apache_dir.iterdir():
            if not project_dir.is_dir():
                continue

            project_name = project_dir.name

            # commit_{number}
            for commit_dir in project_dir.iterdir():
                if not commit_dir.is_dir():
                    continue
                if not commit_dir.name.startswith("commit_"):
                    continue

                commit_number = commit_dir.name

                # id directories
                for id_dir in commit_dir.iterdir():
                    if not id_dir.is_dir():
                        continue

                    try:
                        id_number = int(id_dir.name)
                    except ValueError:
                        continue

                    before_dir = id_dir / "before"
                    if not before_dir.exists():
                        continue

                    target_repo_path = (
                            dataset_programs_root
                            / f"{commit_number}_snapshot"
                            / project_name
                    )

                    results.append(
                        {
                            "antipattern_type": antipattern_type,
                            "project_name": project_name,
                            "commit_number": commit_number,
                            "id": id_number,
                            "antipattern_relation_path": str(before_dir),
                            "target_repo_path": str(target_repo_path),
                        }
                    )

    return results


def record_failed_case(
        *,
        log_path: str,
        antipattern_type: str,
        project_name: str,
        commit_number: str,
        id: str,
        result_folder_name: str,
        output: str,
        exception: Exception,
):
    record = {
        "status": "failed",
        "antipattern_type": antipattern_type,
        "project_name": project_name,
        "commit_number": commit_number,
        "id": id,
        "result_folder_name": result_folder_name,
        "output": output,
        "error_type": type(exception).__name__,
        "error_message": str(exception),
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat(),
    }

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # pairs = collect_antipattern_repo_pairs("/data/sanglei/反模式修复数据集构建")
    # for item in pairs[:3]:
    #     print(item)
    run_batch(antipattern_type_limit="ch", ablation_type="ap_none")
    # run_batch(antipattern_type_limit="awd")
    # run_batch(antipattern_type_limit="mh")
