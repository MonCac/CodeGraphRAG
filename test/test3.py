import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict


def check_duplicate_relations(json_path: str) -> None:
    """
    检查 JSON 文件中是否存在重复关系
    关系重复定义：from_id + to_id + type 相同
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # 读取 JSON 文件
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)  # 假设 JSON 是列表形式，每项是一个关系 dict

    # 统计重复关系
    # 关系 key 由 (from_id, to_id, type) 组成
    rel_keys = [(r['from_id'], r['to_id'], r['type'], str(r['properties'])) for r in data['relationships']]
    counter = Counter(rel_keys)

    # 找出重复的关系
    duplicates = {k: v for k, v in counter.items() if v > 1}

    if duplicates:
        print(f"发现 {len(duplicates)} 种重复关系，共 {sum(duplicates.values())} 条重复关系")
        # 可选：打印前 10 条重复关系
        for i, ((from_id, to_id, rel_type, property), count) in enumerate(duplicates.items()):
            print(f"{i + 1}. from_id={from_id}, to_id={to_id}, type={rel_type}, property={property}, 出现次数={count}")
            if i >= 9:
                break
    else:
        print("未发现重复关系。")


if __name__ == '__main__':
    check_duplicate_relations("D:\智能重构\CodeGraphRAG\codebase_rag\enre\.tmp\my_project_subgraph.json")
