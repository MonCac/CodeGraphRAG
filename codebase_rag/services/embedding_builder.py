from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from codebase_rag.services.llm import create_semantic_model, create_embedding_model
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
    "Module": {"props": ["name", "qualifiedName", "location"], "type": "code_snippet"},
    "Package": {"props": ["name", "qualifiedName"], "type": "description"},
    "Project": {"props": ["name"], "type": "description"},   # 生成 知识图谱 时添加，一般无意义
    "Record": {"props": ["name", "qualifiedName", "location"], "type": "code_snippet"},
    "TypeParameter": {"props": ["name", "location"], "type": "code_snippet"},
    "Variable": {"props": ["name", "location"], "type": "code_snippet"},
}


# ------------------------------------------------------------
def _extract_code_snippet(node: Dict[str, Any], start: int, end: int) -> str:
    """根据 location 或 startLine/endLine 提取源码"""
    props = node.get("properties", {})
    loc = props.get("location", props)
    path = loc.get("binPath") or props.get("additionalBin", {}).get("binPath")
    if not path or not Path(path).exists():
        return ""
    start = start - 1 if start else 0
    end = end if end else start + 1
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    return "".join(lines[start:end])


def _read_entire_file(node: Dict[str, Any]) -> str:
    props = node.get("properties", {})
    path = props.get("additionalBin", {}).get("binPath")
    if not path or not Path(path).exists():
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class SemanticGenerator:
    def __init__(self, client: Any):
        self.client = client
        self.semantic_cache: Dict[int, str] = {}

    def generate_node_semantics(
            self, node_id: int, nodes: Dict[int, Dict[str, Any]],
            children_map: Dict[int, List[int]]
    ) -> str:
        if node_id in self.semantic_cache:
            return self.semantic_cache[node_id]

        node = nodes[node_id]
        labels = node.get("labels", [])
        props = node.get("properties", {})

        # 递归生成子节点摘要
        child_texts = [self.generate_node_semantics(c, nodes, children_map)
                       for c in children_map.get(node_id, [])]
        child_summary = "\n".join(child_texts) if child_texts else ""

        # 找到匹配的 label 处理规则
        handler = next((NODE_HANDLERS[l] for l in labels if l in NODE_HANDLERS), None)
        node_type = handler["type"] if handler else "description"

        # 构建上下文
        code_snippet = ""
        try:
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
            # description 类型不提取源码
        except Exception as e:
            logger.warning(f"源码提取失败（node_id={node_id}）：{e}")

        # 构建 prompt，可替换为 prompts.py 中模板
        prompt = f"""
    你是一个代码语义分析助手。
    当前节点：{labels[0]} {props.get('name', '')}
    节点属性：
    {json.dumps(props, ensure_ascii=False)}
    {f"代码片段：\n{code_snippet}" if code_snippet else ""}
    子节点摘要：
    {child_summary}

    请用一句话总结当前节点在代码中的功能或作用，不超过50字。
    """
        try:
            # 调用本地模型生成摘要
            result = self.client.generate(prompt)
            # 假设本地模型返回对象有 output 属性
            desc = getattr(result, "output", str(result)).strip()
        except Exception as e:
            logger.warning(f"LLM 生成描述失败（node_id={node_id}）：{e}")
            desc = f"{labels[0]} {props.get('name', '')}"

        self.semantic_cache[node_id] = desc
        return desc


class EmbeddingBuilder:
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.client = create_semantic_model()
        self.embedding_model = create_embedding_model()
        self.chroma_dir = self.output_dir / "chroma"
        self.collection_name = "nodes"

        # 初始化 LangChain Chroma
        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.chroma_dir),
        )

        # 初始化语义生成器
        self.semantic_generator = SemanticGenerator(client=self.client)

    # ------------------------------------------------------------
    def build_hierarchical_embeddings(self, graph_data: Dict[str, Any]) -> None:
        """
        根据知识图谱构建节点层级语义 embedding 并保存到 Chroma。
        """
        nodes = {n["node_id"]: n for n in graph_data.get("nodes", [])}
        relationships = graph_data.get("relationships", [])

        # 构建 parent -> children map
        children_map: Dict[int, List[int]] = {}
        for r in relationships:
            children_map.setdefault(r["from_id"], []).append(r["to_id"])

        texts, metadatas, ids = [], [], []

        for node_id in nodes:
            desc = self.semantic_generator.generate_node_semantics(
                node_id, nodes, children_map
            )
            node = nodes[node_id]
            texts.append(desc)
            ids.append(str(node_id))
            metadatas.append({
                "node_id": node_id,
                "labels": node.get("labels"),
                "properties": node.get("properties"),
                "semantic_description": desc
            })

        # 保存到 LangChain Chroma
        self.vectordb.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        self.vectordb.persist()
        logger.success("✅ 知识图谱式语义 embedding 已保存到 LangChain Chroma。")

    def _get_embedding(self, text: str) -> List[float]:
        """
        保留单独调用 OpenAI embedding 的方法（可选）。
        LangChain Chroma 内部也会调用 embedding_function。
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding 生成失败: {e}")
            return [0.0] * 1536
