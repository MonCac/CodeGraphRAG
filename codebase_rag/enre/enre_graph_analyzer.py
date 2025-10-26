import datetime
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from loguru import logger
from codebase_rag.enre.loader import ENRELoader
from codebase_rag.main import resolve_output_path


class ENREGraphAnalyzer:
    """根据反模式结果递归搜索项目依赖图，提取相关子图"""

    def __init__(self, project_path: str | Path, antipattern_path: str | Path):
        self.project_path = Path(project_path).resolve()
        self.antipattern_path = Path(antipattern_path).resolve()
        self.project_loader = ENRELoader(self.project_path)
        self.antipattern_loader = ENRELoader(self.antipattern_path)
        self.nodes: List[Dict[str, Any]] = []
        self.relationships: List[Dict[str, Any]] = []

    def generate_subgraph(self, max_depth: int | None = None, max_nodes: int = 100000):
        """
        生成基于反模式节点的项目子图（保持原始 ENRE 节点/关系格式）。
        同时在日志输出每层节点数量，并去重关系。

        参数:
            max_depth: 最大递归层数（None 表示不限制，但内部硬限制为100）
            max_nodes: 节点数上限，用于提前停止搜索

        返回:
            dict: 包含 'nodes', 'relationships', 'metadata' 的子图
        """

        # 1️⃣ 运行 ENRE 分析
        project_nodes, project_rels = self.project_loader.get_nodes_and_relationships()
        antipattern_nodes, _ = self.antipattern_loader.get_nodes_and_relationships()

        # 2️⃣ 构建项目图的邻接表（无向遍历）
        adjacency = {}
        for rel in project_rels:
            adjacency.setdefault(rel["from_id"], set()).add(rel["to_id"])
            adjacency.setdefault(rel["to_id"], set()).add(rel["from_id"])

        # 3️⃣ 确定起点节点（反模式节点）
        start_nodes = {n["node_id"] for n in antipattern_nodes}
        visited = set()
        queue = [(nid, 0) for nid in start_nodes]

        # 4️⃣ BFS 遍历，内部维护 depth，每个节点第一次访问就记录深度
        MAX_ALLOWED_DEPTH = 2  # 硬限制，防止过深遍历
        node_depths: dict[str, int] = {}  # node_id -> depth

        while queue:
            node_id, depth = queue.pop(0)

            # 已访问过的节点跳过
            if node_id in node_depths:
                continue

            # 超过深度限制的节点跳过
            if (max_depth is not None and depth >= max_depth) or depth >= MAX_ALLOWED_DEPTH:
                continue

            # 记录节点深度
            node_depths[node_id] = depth
            visited.add(node_id)

            for neighbor in adjacency.get(node_id, []):
                if neighbor not in node_depths:
                    queue.append((neighbor, depth + 1))

            # 节点数上限
            if len(visited) >= max_nodes:
                logger.warning(f"Node limit ({max_nodes}) reached, stopping traversal.")
                break

        # 5️⃣ 构建子图（原始节点和关系，不添加 depth 字段）
        project_node_map = {n["node_id"]: n for n in project_nodes}
        sub_nodes = [project_node_map[nid] for nid in visited if nid in project_node_map]

        # 5️⃣-1 去重关系
        seen = set()
        unique_sub_rels = []
        for rel in project_rels:
            if rel["from_id"] in visited and rel["to_id"] in visited:
                key = (rel["from_id"], rel["to_id"], rel.get("type"))
                if key not in seen:
                    seen.add(key)
                    unique_sub_rels.append(rel)

        sub_rels = unique_sub_rels

        # 6️⃣ 打印每层节点数量
        from collections import Counter
        depth_counter = Counter(node_depths.values())
        depth_info = ", ".join(f"depth {d}: {c} nodes" for d, c in sorted(depth_counter.items()))
        logger.info(f"Subgraph node distribution by depth: {depth_info}")

        # 7️⃣ 返回结果
        self.nodes = sub_nodes
        self.relationships = sub_rels
        result = {
            "nodes": sub_nodes,
            "relationships": sub_rels,
            "metadata": {
                "project_repo": str(self.project_path),
                "antipattern_repo": str(self.antipattern_path),
                "total_nodes": len(sub_nodes),
                "total_relationships": len(sub_rels),
                "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "max_depth": max_depth,
                "max_nodes": max_nodes,
                "max_allowed_depth": MAX_ALLOWED_DEPTH
            },
        }

        logger.info(
            f"Generated subgraph with {len(sub_nodes)} nodes, "
            f"{len(sub_rels)} relationships."
        )

        return result

    def save_subgraph(self, output_path: str | Path, **kwargs):
        """生成并保存子图 JSON"""
        subgraph = self.generate_subgraph(**kwargs)
        output_path = resolve_output_path(output_path)
        output_path = Path(output_path)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(subgraph, f, ensure_ascii=False, indent=2)

        logger.info(f"Subgraph saved to {output_path}")
        return output_path

    def get_nodes_and_relationships(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """返回当前分析器中的节点和关系"""
        return self.nodes, self.relationships
