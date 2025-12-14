# codebase_rag/enre/loader.py
import datetime
import shutil
import subprocess
import chardet
from pathlib import Path
from loguru import logger
import orjson, json
from typing import Any, List, Dict, Tuple
from codebase_rag.enre.relation_type import RelationType


class ENRELoader:
    """Loads ENRE JSON files and converts them into nodes and relationships for Memgraph."""

    def __init__(self, repo_path: str | Path, output_dir: Path | str):
        self.repo_path = Path(repo_path).resolve()
        self.repo_name = self.repo_path.name
        self.tmp_dir = Path(output_dir).resolve()
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.json_path: Path | None = None
        self.data: Dict[str, Any] = {}
        self.nodes: List[Dict[str, Any]] = []
        self.relationships: List[Dict[str, Any]] = []

    def run_enre_analysis(self) -> None:
        """Run ENRE analysis using the enre_java.jar on the repo_path."""
        jar_path = Path("lib/enre_java.jar").resolve()
        if not jar_path.exists():
            raise FileNotFoundError(f"ENRE JAR not found at: {jar_path}")

        # 执行命令
        cmd = ["java", "-jar", str(jar_path), "java", str(self.repo_path), self.repo_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"cmd: {cmd}")
        if result.returncode != 0:
            raise RuntimeError(
                f"ENRE analysis failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )

        # 分析完成后生成的文件路径
        output_folder = Path(f"{self.repo_name}-enre-out").resolve()
        output_json = output_folder / f"{self.repo_name}-out.json"
        if not output_json.exists():
            raise FileNotFoundError(f"Expected ENRE output JSON not found: {output_json}")

        # 移动到 tmp 目录
        target_json = self.tmp_dir / f"{self.repo_name}-out.json"
        shutil.move(str(output_json), str(target_json))
        self.json_path = target_json
        logger.info(f"ENRE output moved to: {self.json_path}")

        # 如果原目录为空，则删除它
        try:
            output_folder.rmdir()  # 只能删除空文件夹
            logger.info(f"Removed empty folder: {output_folder}")
        except OSError:
            logger.debug(f"Folder not empty, not removed: {output_folder}")

        # 加载 JSON 数据
        self.data = self._load_json()

    def _load_json(self) -> Dict[str, Any]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"ENRE JSON file not found: {self.json_path}")

        # 只读取前 4096 字节进行编码检测
        with self.json_path.open("rb") as f:
            head = f.read(4096)
            detected = chardet.detect(head)
            enc = detected.get("encoding") or "utf-8"

        try:
            with self.json_path.open("r", encoding=enc, errors="replace") as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON with detected encoding '{enc}': {e}") from e

    def parse_entities(self) -> List[Dict[str, Any]]:
        """Convert ENRE entities to node dicts: {node_id, labels, properties}."""
        entities = self.data.get("variables", [])
        nodes = []
        for ent in entities:
            if ent.get("external", False):
                continue
            node = {
                "node_id": ent["id"],  # 改为 node_id
                "labels": [ent["category"].replace(" ", "")],   # category 放到 labels 列表里并去掉空格，解决"Enum Constant"、"Type Parameter"无法存储 Memgraph 问题
                "properties": {k: v for k, v in ent.items() if k not in ("id", "category")}
            }
            nodes.append(node)
        self.nodes = nodes
        return nodes

    def parse_relationships(self) -> List[Dict[str, Any]]:
        """Convert ENRE relations to relationship dicts: {from_id, to_id, type, properties}."""
        relations = self.data.get("cells", [])
        rels = []

        for rel in relations:
            from_id = rel["src"]
            to_id = rel["dest"]
            # type 取 values 中第一个字段（除 loc）对应的 key，且值为 1
            values = rel.get("values", {})
            rel_type = None
            for k, v in values.items():
                if k != "loc" and v == 1:
                    rel_type = k
                    break
            if rel_type is None:
                rel_type = "Unknown"

            if rel_type not in RelationType._member_names_:
                rel_type_enum = "Unknown"
            else:
                rel_type_enum = rel_type

            relationship = {
                "from_id": from_id,
                "to_id": to_id,
                "type": rel_type_enum,
                "properties": {k: v for k, v in values.items() if k != rel_type}  # 除 type 外的其他信息存 properties
            }
            rels.append(relationship)

        self.relationships = rels
        return rels

    def get_nodes_and_relationships(self, output_dir: Path | str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Parse and return both nodes and relationships."""
        self.run_enre_analysis()
        self.parse_entities()
        self.parse_relationships()
        self.save_json(output_dir)
        return self.nodes, self.relationships

    def save_json(self, output_dir: Path | str) -> str:
        """将 ENRE 结果保存为 JSON 文件，路径固定为 output_dir/{repo_name}-graph.json。"""
        data = self.to_json_dict()
        repo_name = self.repo_path.name
        logger.info(f"start ENRE GRAPH SAVE JSON")

        output_path = Path(output_dir) / f"{repo_name}-graph.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("wb") as f:  # 注意是二进制写入
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

        logger.info(f"ENRE results saved to {output_path}")
        return str(output_path)

    def to_json_dict(self) -> Dict[str, Any]:
        """组织 ENRE 结果为 JSON dict 格式。"""
        if not self.nodes or not self.relationships:
            # 确保先解析过
            self.get_nodes_and_relationships()

        return {
            "nodes": self.nodes,
            "relationships": self.relationships,
            "metadata": {
                "total_nodes": len(self.nodes),
                "total_relationships": len(self.relationships),
                "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
            },
        }

