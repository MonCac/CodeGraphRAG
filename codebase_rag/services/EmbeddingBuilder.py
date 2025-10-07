from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from loguru import logger
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings


class EmbeddingBuilder:
    def __init__(self, output_dir: Path):
        """
        Args:
            output_dir: Chroma 持久化存储目录
        """
        self.output_dir = Path(output_dir)
        self.client = OpenAI()
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.chroma_dir = self.output_dir / "chroma"
        self.collection_name = "nodes"

        self.vectordb = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            persist_directory=str(self.chroma_dir),
        )

    # ------------------------------------------------------------
    def build_hierarchical_embeddings(self, graph_data: Dict[str, Any]) -> None:
        """
        根据知识图谱构建节点层级语义 embedding 并保存到 Chroma。
        """
        nodes = {n["node_id"]: n for n in graph_data.get("nodes", [])}
        relationships = graph_data.get("relationships", [])

        # 建立 parent -> children 映射
        children_map: Dict[int, List[int]] = {}
        for r in relationships:
            parent_id = r["from_id"]
            child_id = r["to_id"]
            children_map.setdefault(parent_id, []).append(child_id)

        # 递归缓存节点语义，避免重复生成
        semantic_cache: Dict[int, str] = {}

        def generate_node_semantics(node_id: int) -> str:
            if node_id in semantic_cache:
                return semantic_cache[node_id]

            node = nodes[node_id]
            labels = node.get("labels", [])
            label_str = ", ".join(labels)
            props = node.get("properties", {})

            # 收集子节点语义
            child_texts = []
            for child_id in children_map.get(node_id, []):
                child_texts.append(generate_node_semantics(child_id))
            child_summary = "\n".join(child_texts) if child_texts else ""

            # 构建 prompt
            prompt = f"""
你是一个代码语义分析助手。
当前节点类型：{label_str}
节点属性：{json.dumps(props, ensure_ascii=False)}

子节点摘要：
{child_summary}

请生成一句话总结当前节点在代码中的功能或作用，简洁明了，不超过50字。
"""
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                desc = response.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"LLM 生成描述失败（node_id={node_id}）：{e}")
                desc = f"{label_str} {props.get('name', '')}"

            semantic_cache[node_id] = desc
            return desc

        # 遍历所有节点生成语义并构建 Chroma 数据
        texts, metadatas, ids = [], [], []
        for node_id in nodes:
            desc = generate_node_semantics(node_id)
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

    # ------------------------------------------------------------
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
