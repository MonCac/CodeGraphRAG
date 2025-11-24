import os
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from codebase_rag.enre.relation_type import RelationType
import mgclient
from loguru import logger


class MemgraphIngestor:
    """Handles all communication and query execution with the Memgraph database."""

    def __init__(self, host: str, port: int, batch_size: int = 1000):
        self._host = host
        self._port = port
        self.batch_size = batch_size
        self.conn: mgclient.Connection | None = None
        self.node_buffer: list[tuple[str, dict[str, Any]]] = []
        self.relationship_buffer: list[tuple[tuple, str, tuple, dict | None]] = []
        self.unique_constraints = {
            "Enum": "id",
            "TypeParameter": "id",
            "Variable": "id",
            "Annotation Member": "id",
            "Class": "id",
            "Package": "id",
            "Method": "id",
            "File": "id",
            "Interface": "id",
            "Annotation": "id",
        }

    def __enter__(self) -> "MemgraphIngestor":
        logger.info(f"Connecting to Memgraph at {self._host}:{self._port}...")
        self.conn = mgclient.connect(host=self._host, port=self._port)
        self.conn.autocommit = True
        logger.info("Successfully connected to Memgraph.")
        return self

    def __exit__(
            self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any
    ) -> None:
        if exc_type:
            logger.error(
                f"An exception occurred: {exc_val}. Flushing remaining items...",
                exc_info=True,
            )
        self.flush_all()
        if self.conn:
            self.conn.close()
            logger.info("\nDisconnected from Memgraph.")

    def _execute_query(self, query: str, params: dict[str, Any] | None = None) -> list:
        if not self.conn:
            raise ConnectionError("Not connected to Memgraph.")
        params = params or {}
        cursor = None
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            if not cursor.description:
                return []
            column_names = [desc.name for desc in cursor.description]
            return [dict(zip(column_names, row)) for row in cursor.fetchall()]
        except Exception as e:
            if (
                    "already exists" not in str(e).lower()
                    and "constraint" not in str(e).lower()
            ):
                logger.error(f"!!! Cypher Error: {e}")
                logger.error(f"    Query: {query}")
                logger.error(f"    Params: {params}")
            raise
        finally:
            if cursor:
                cursor.close()

    def _execute_batch(self, query: str, params_list: list[dict[str, Any]]) -> None:
        if not self.conn or not params_list:
            return
        cursor = None
        try:
            cursor = self.conn.cursor()
            batch_query = f"UNWIND $batch AS row\n{query}"
            logger.debug(batch_query)
            cursor.execute(batch_query, {"batch": params_list})
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.error(f"!!! Batch Cypher Error: {e}")
        finally:
            if cursor:
                cursor.close()

    def clean_database2(
            self,
            volume_name: str = "mg_lib",
            max_retries: int = 30,
            retry_interval: float = 1.0,
    ) -> None:
        """
        完全清空 Docker Memgraph 数据库，并重置 node_id。
        自动查找 Memgraph 容器、停止、清空卷内容、重启，并循环重连。
        适合在 with 上下文中直接调用。

        Args:
            volume_name: Docker 卷名
            max_retries: 最大重试连接次数
            retry_interval: 每次重试等待秒数
        """
        logger.info("--- Cleaning Docker Memgraph database (node_id will reset) ---")
        self._execute_query("MATCH (n) DETACH DELETE n;")

        # 1️⃣ 关闭现有连接
        if self.conn:
            try:
                self.conn.close()
                logger.info("Closed current Memgraph connection.")
            except Exception:
                pass
            self.conn = None

        # 2️⃣ 查找 Memgraph 容器
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "ancestor=memgraph/memgraph-platform", "--format", "{{.Names}}"],
                capture_output=True, text=True, check=True
            )
            containers = result.stdout.strip().splitlines()
            if not containers:
                logger.error("No running Memgraph container found.")
                return
            container_name = containers[0]
            logger.info(f"Found Memgraph container: {container_name}")
        except Exception as e:
            logger.error(f"Failed to list Docker containers: {e}")
            return

        # 3️⃣ 停止容器
        try:
            subprocess.run(["docker", "stop", container_name], check=True)
            logger.info(f"Container '{container_name}' stopped.")
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return

        # 4️⃣ 获取卷挂载路径并清空内容
        try:
            inspect = subprocess.run(
                ["docker", "volume", "inspect", "--format", "{{.Mountpoint}}", volume_name],
                capture_output=True, text=True, check=True
            )
            mount_path = inspect.stdout.strip()
            if os.path.exists(mount_path):
                # 清空卷内容
                for entry in os.listdir(mount_path):
                    entry_path = os.path.join(mount_path, entry)
                    if os.path.isdir(entry_path):
                        shutil.rmtree(entry_path)
                    else:
                        os.remove(entry_path)
                logger.info(f"Cleared contents of Docker volume '{volume_name}' at '{mount_path}'.")
            else:
                logger.warning(f"Mount path '{mount_path}' does not exist. Volume may be empty.")
        except Exception as e:
            logger.error(f"Failed to clear Docker volume contents: {e}")
            return

        # 5️⃣ 启动容器
        try:
            subprocess.run(["docker", "start", container_name], check=True)
            logger.info(f"Container '{container_name}' restarted.")
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return

        # 6️⃣ 循环重试连接
        logger.info("Waiting for Memgraph to be ready...")
        for attempt in range(1, max_retries + 1):
            try:
                self.conn = mgclient.connect(host=self._host, port=self._port)
                self.conn.autocommit = True
                logger.info(f"Reconnected to Memgraph on attempt {attempt}.")
                break
            except Exception:
                logger.debug(f"Attempt {attempt} failed, retrying in {retry_interval}s...")
                time.sleep(retry_interval)
        else:
            logger.error(f"Failed to reconnect to Memgraph after {max_retries} attempts.")
            self.conn = None

    def clean_database(self) -> None:
        logger.info("--- Cleaning database with DROP GRAPH ---")
        try:
            cursor = self.conn.cursor()

            # 1. 查询当前存储模式
            cursor.execute("SHOW STORAGE INFO;")
            storage_info = cursor.fetchall()

            # 2. 查找 storage_mode
            current_mode = None
            for key, value in storage_info:
                if key == "storage_mode":
                    current_mode = value
                    break

            logger.info(f"Current storage mode: {current_mode}")

            # 3. 如果不是 IN_MEMORY_ANALYTICAL，切换
            if current_mode != "IN_MEMORY_ANALYTICAL":
                logger.info(f"Switching storage mode from {current_mode} to IN_MEMORY_ANALYTICAL.")
                cursor.execute("STORAGE MODE IN_MEMORY_ANALYTICAL;")

            current_mode = None
            for key, value in storage_info:
                if key == "storage_mode":
                    current_mode = value
                    break

            # 4. 执行 DROP GRAPH
            cursor.execute("DROP GRAPH;")
            logger.info("DROP GRAPH executed successfully.")

            cursor.close()
        except Exception as e:
            logger.error(f"Failed to clean database: {e}")

    def clean_database_before(self) -> None:
        logger.info("--- Cleaning database... ---")
        self._execute_query("MATCH (n) DETACH DELETE n;")
        logger.info("--- Database cleaned. ---")

    def ensure_constraints(self) -> None:
        """Ensure unique constraints for all node labels in Memgraph."""
        logger.info("Ensuring constraints...")
        for label, prop in self.unique_constraints.items():
            # 将标签中的空格替换为下划线，生成合法 Cypher 标签名
            safe_label = label.replace(" ", "_")
            try:
                self._execute_query(
                    f"CREATE CONSTRAINT ON (n:{safe_label}) ASSERT n.{prop} IS UNIQUE;"
                )
            except Exception:
                # 已存在的约束会抛异常，可以忽略
                pass
        logger.info("Constraints checked/created.")

    def ensure_node_batch(self, label: str, properties: dict[str, Any]) -> None:
        """Adds a node to the buffer."""
        self.node_buffer.append((label, properties))

    def ensure_relationship_batch(
            self,
            from_spec: tuple[str, str, Any],
            rel_type: str,
            to_spec: tuple[str, str, Any],
            properties: dict[str, Any] | None = None,
    ) -> None:
        """Adds a relationship to the buffer."""
        from_label, from_key, from_val = from_spec
        to_label, to_key, to_val = to_spec
        self.relationship_buffer.append(
            (
                (from_label, from_key, from_val),
                rel_type,
                (to_label, to_key, to_val),
                properties,
            )
        )

    def flush_nodes(self) -> None:
        """Flushes buffered nodes (from ENRE JSON) to Memgraph."""
        if not self.node_buffer:
            return

        # flush前导出全量图数据，获取节点和关系数
        graph_before = self.export_graph_to_dict()
        logger.info(
            f"[flush_nodes] Before flush: {graph_before['metadata']['total_nodes']} nodes, {graph_before['metadata']['total_relationships']} relationships.")

        nodes_by_label = defaultdict(list)
        for label, props in self.node_buffer:
            nodes_by_label[label].append(props)

        for label, props_list in nodes_by_label.items():
            if not props_list:
                continue

            id_key = "id"  # 用 id 做唯一标识

            prop_keys = [key for key in props_list[0].keys() if key != id_key]
            set_clause = ", ".join([f"n.{key} = row.{key}" for key in prop_keys])
            query = (
                f"MERGE (n:{label} {{{id_key}: row.{id_key}}}) "
                f"ON CREATE SET {set_clause} ON MATCH SET {set_clause}"
            )
            self._execute_batch(query, props_list)

        logger.info(f"Flushed {len(self.node_buffer)} nodes.")
        self.node_buffer.clear()

        # flush后再导出全量图数据
        graph_after = self.export_graph_to_dict()
        logger.info(
            f"[flush_nodes] After flush: {graph_after['metadata']['total_nodes']} nodes, {graph_after['metadata']['total_relationships']} relationships.")

    def flush_relationships(self) -> None:
        if not self.relationship_buffer:
            return

        graph_before = self.export_graph_to_dict()
        logger.info(
            f"[flush_relationships] Before flush: {graph_before['metadata']['total_nodes']} nodes, {graph_before['metadata']['total_relationships']} relationships.")

        rels_by_pattern = defaultdict(list)
        for from_node, rel_type, to_node, props in self.relationship_buffer:
            pattern = (from_node[0], from_node[1], rel_type, to_node[0], to_node[1])
            rels_by_pattern[pattern].append(
                {"from_val": from_node[2], "to_val": to_node[2], "props": props or {}}
            )

        for pattern, params_list in rels_by_pattern.items():
            from_label, from_key, rel_type, to_label, to_key = pattern
            query = (
                f"MATCH (a:{from_label} {{{from_key}: row.from_val}}), "
                f"(b:{to_label} {{{to_key}: row.to_val}})\n"
                f"MERGE (a)-[r:{rel_type}]->(b)"
            )
            if any(p["props"] for p in params_list):
                query += "\nSET r += row.props"

            self._execute_batch(query, params_list)

        logger.info(f"Flushed {len(self.relationship_buffer)} relationships.")
        self.relationship_buffer.clear()

        graph_after = self.export_graph_to_dict()
        logger.info(
            f"[flush_relationships] After flush: {graph_after['metadata']['total_nodes']} nodes, {graph_after['metadata']['total_relationships']} relationships.")

    def flush_all(self) -> None:
        logger.info("--- Flushing all pending writes to database... ---")
        self.flush_nodes()
        self.flush_relationships()
        logger.info("--- Flushing complete. ---")

    def fetch_all(self, query: str, params: dict[str, Any] | None = None) -> list:
        """Executes a query and fetches all results."""
        logger.debug(f"Executing fetch query: {query} with params: {params}")
        return self._execute_query(query, params)

    def execute_write(self, query: str, params: dict[str, Any] | None = None) -> None:
        """Executes a write query without returning results."""
        logger.debug(f"Executing write query: {query} with params: {params}")
        self._execute_query(query, params)

    def export_graph_to_dict(self) -> dict[str, Any]:
        """Export the entire graph as a dictionary with nodes and relationships."""
        logger.info("Exporting graph data...")

        # Get all nodes with their labels and properties, export user-defined 'id' property instead of internal id
        nodes_query = """
        MATCH (n)
        RETURN n.id AS node_id, labels(n) AS labels, properties(n) AS properties
        """
        nodes_data = self.fetch_all(nodes_query)

        # Get all relationships with their types and properties, using user-defined 'id' properties on nodes
        relationships_query = """
        MATCH (a)-[r]->(b)
        RETURN a.id AS from_id, b.id AS to_id, type(r) AS type, properties(r) AS properties
        """
        relationships_data = self.fetch_all(relationships_query)

        graph_data = {
            "nodes": nodes_data,
            "relationships": relationships_data,
            "metadata": {
                "total_nodes": len(nodes_data),
                "total_relationships": len(relationships_data),
                "exported_at": self._get_current_timestamp(),
            },
        }

        logger.info(
            f"Exported {len(nodes_data)} nodes and {len(relationships_data)} relationships"
        )
        return graph_data

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(UTC).isoformat()
