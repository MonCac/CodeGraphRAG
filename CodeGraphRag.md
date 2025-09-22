# CodeGraphRag

## Memgraph

首先为了连接memgraph，下载它

## 安装和启动 Memgraph

### 使用 Docker

```bash
# 拉取 Memgraph 镜像
docker pull memgraph/memgraph-platform

# 启动 Memgraph   未进行本地目录与虚拟目录的存储映射
docker run -it -p 7687:7687 -p 3000:3000 -p 7444:7444 -v mg_lib:/var/lib/memgraph memgraph/memgraph-platform
```



Memgraph Lab

```bash
http://localhost:3000/

# 查询所有 nodes 和 relations
MATCH (n)-[r]->(m)
RETURN n, r, m


    def flush_relationships(self) -> None:
        """Flushes buffered relationships (from ENRE JSON) to Memgraph."""
        if not self.relationship_buffer:
            return

        # 直接创建每条关系，不做合并
        for from_node, rel_type, to_node, props in self.relationship_buffer:
            query = (
                f"MATCH (a {{id: $from_val}}), (b {{id: $to_val}}) "
                f"CREATE (a)-[r:{rel_type}]->(b) "
                f"SET r += $props"
            )
            params = {"from_val": from_node[2], "to_val": to_node[2], "props": props or {}}
            self._execute_query(query, params)

        logger.info(f"Flushed {len(self.relationship_buffer)} relationships.")
        self.relationship_buffer.clear()
```





## Ollama

如果使用的是本地Ollama，需要启动Ollama

### 启动 Ollama

```bash
ollama serve
```

此处需要注意，ollama的模型必须支持Tools. 如 `gemma3:1b` 不可行， `qwen3:0.6b` 可行
