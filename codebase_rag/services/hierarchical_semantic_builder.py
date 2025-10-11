from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from openai import OpenAI
from loguru import logger
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from codebase_rag.prompts import get_node_semantic_prompt
from codebase_rag.services.llm import create_semantic_model, create_embedding_model
from tqdm import tqdm

# ------------------------------------------------------------
# 节点类型处理配置
NODE_HANDLERS = {
    "Annotation": {"props": ["name", "location"], "type": "code_snippet"},
    "AnnotationMember": {"props": ["name", "location"], "type": "code_snippet"},
    "Class": {"props": ["name", "qualifiedName", "location"], "type": "code_snippet"},
    "Enum": {"props": ["name", "qualifiedName", "location"], "type": "code_snippet"},
    "EnumConstant": {"props": ["name", "parentId"], "type": "code_snippet_parent"},  # 需使用 parent location
    "File": {"props": ["name", "additionalBin"], "type": "file_code"},  # 整个文件
    "Interface": {"props": ["name", "qualifiedName", "location"], "type": "code_snippet"},
    "Method": {"props": ["name", "qualifiedName", "startLine", "endLine"], "type": "code_snippet"},
    "Module": {"props": ["name", "qualifiedName", "location"], "type": "description"},
    "Package": {"props": ["name", "qualifiedName"], "type": "description"},
    "Project": {"props": ["name"], "type": "description"},  # 生成 知识图谱 时添加，一般无意义
    "Record": {"props": ["name", "qualifiedName", "location"], "type": "code_snippet"},
    "TypeParameter": {"props": ["name", "location"], "type": "code_snippet"},
    "Variable": {"props": ["name", "location"], "type": "code_snippet"},
}


# ------------------------------------------------------------
def _extract_code_snippet(node: Dict[str, Any], start: int, end: int) -> str:
    """根据 location 或 startLine/endLine 提取源码"""
    props = node.get("properties", {})
    loc = props.get("location", props)

    # 获取路径和文件名
    bin_path = loc.get("binPath") or props.get("additionalBin", {}).get("binPath")
    file_name = props.get("File")

    # 拼接路径
    if bin_path and file_name:
        path = Path(bin_path) / file_name
    elif bin_path:
        path = Path(bin_path)
    else:
        return ""
    if not path or not Path(path).exists():
        return ""
    start = start - 1 if start else 0
    end = end if end else start + 1
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[start:end])


def _read_entire_file(node: Dict[str, Any]) -> str:
    props = node.get("properties", {})
    bin_path = props.get("additionalBin", {}).get("binPath")
    file_name = props.get("name")

    if bin_path and file_name:
        path = Path(bin_path) / file_name
    elif bin_path:
        path = Path(bin_path)
    else:
        return ""
    if not path or not Path(path).exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ------------------------------------------------------------
# SemanticGenerator
# ------------------------------------------------------------
class SemanticGenerator:
    def __init__(self, client: Any):
        self.client = client
        self.semantic_cache: Dict[int, str] = {}

    def generate_node_semantics(
            self,
            node_id: int,
            nodes: Dict[int, Dict[str, Any]],
            children_map: Dict[int, List[tuple]],
            store_callback: Optional[Callable[[int, str], None]] = None,
            visited: Optional[set] = None
    ) -> str:
        # 初始化访问记录集
        if visited is None:
            visited = set()

        # 检查是否已访问（防止循环依赖造成无限递归）
        if node_id in visited:
            tqdm.write(f"⚠️ 检测到 relation 环：node_id={node_id} 已访问，跳过递归。")
            return f"[循环引用节点 {node_id}]"

        # 将当前节点标记为已访问
        visited.add(node_id)

        # 如果有缓存，直接返回
        if node_id in self.semantic_cache:
            return self.semantic_cache[node_id]

        node = nodes[node_id]
        labels = node.get("labels", [])
        props = node.get("properties", {})

        # 递归生成子节点摘要，并加入 relation
        child_entries = children_map.get(node_id, [])
        child_texts = []
        for child_id, relation in child_entries:
            child_desc = self.generate_node_semantics(
                child_id, nodes, children_map, store_callback, visited.copy()
            )
            child_texts.append(f"{relation}: {child_desc}")
        child_summary = "\n".join(child_texts) if child_texts else ""

        # 构建上下文
        code_snippet = ""
        try:
            node_type = "description"
            handler = next((NODE_HANDLERS[l] for l in labels if l in NODE_HANDLERS), None)
            if handler:
                node_type = handler["type"]

            if node_type == "code_snippet":
                location = props.get("location", {})
                start, end = location.get("startLine"), location.get("endLine")
                code_snippet = _extract_code_snippet(node, start, end)
            elif node_type == "code_snippet_parent":
                parent_id = props.get("parentId")
                parent_node = nodes.get(parent_id)
                if parent_node:
                    location = parent_node.get("properties", {}).get("location", {})
                    start, end = location.get("startLine"), location.get("endLine")
                    code_snippet = _extract_code_snippet(parent_node, start, end)
            elif node_type == "file_code":
                code_snippet = _read_entire_file(node)
        except Exception as e:
            logger.warning(f"源码提取失败（node_id={node_id}）：{e}")

        # 构建 prompt
        prompt = get_node_semantic_prompt(labels, props, code_snippet, child_summary)
        try:
            result = self.client.run_sync(prompt)
            desc = getattr(result, "output", str(result)).strip()
        except Exception as e:
            logger.warning(f"LLM 生成描述失败（node_id={node_id}）：{e}")
            desc = f"{labels[0]} {props.get('name', '')}"

        # 缓存语义
        self.semantic_cache[node_id] = desc

        # 立即存储
        if store_callback:
            store_callback(node_id, desc)

        return desc


# ------------------------------------------------------------
# HierarchicalSemanticBuilder
# ------------------------------------------------------------
class HierarchicalSemanticBuilder:
    def __init__(self):
        self.client = create_semantic_model()
        self.semantic_generator = SemanticGenerator(self.client)

    def build_node_semantics(
            self,
            graph_data: Dict[str, Any]
    ) -> Dict[str, List]:
        """
        构建整棵节点树的语义描述，并在每生成一个节点时立即存储。
        返回 texts, ids, metadatas
        """
        nodes = {n["node_id"]: n for n in graph_data.get("nodes", [])}
        relationships = graph_data.get("relationships", [])

        # 构建 parent -> children map
        children_map: Dict[int, List[tuple]] = {}
        for r in relationships:
            children_map.setdefault(r["from_id"], []).append(
                (r["to_id"], r["type"])
            )

        texts, ids, metadatas = [], [], []

        # 定义存储回调
        def store_callback(node_id: int, desc: str):
            node = nodes[node_id]
            texts.append(desc)
            ids.append(str(node_id))
            metadatas.append({
                "node_id": node_id,
                "labels": node.get("labels"),
                "properties": node.get("properties"),
                "semantic_description": desc
            })

        # 找到所有根节点（没有父节点的节点）
        child_ids = {child_id for clist in children_map.values() for child_id, _ in clist}
        root_nodes = [nid for nid in nodes if nid not in child_ids]

        total_nodes = len(nodes)
        logger.info(f" 总节点数 {total_nodes}，开始递归生成语义...")

        # 使用进度条
        with tqdm(total=len(nodes), desc="⏳ 生成节点语义中", unit="node") as pbar:
            visited_nodes = set()

            def wrapped_store_callback(node_id: int, desc: str):
                if node_id not in visited_nodes:
                    visited_nodes.add(node_id)
                    pbar.update(1)
                store_callback(node_id, desc)

            for node_id in root_nodes:
                self.semantic_generator.generate_node_semantics(
                    node_id, nodes, children_map, wrapped_store_callback
                )

        return {
            "texts": texts,
            "ids": ids,
            "metadatas": metadatas
        }