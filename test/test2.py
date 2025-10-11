import json
from pathlib import Path

# === 修改为你的图文件路径 ===
graph_path = Path(r"D:\智能重构\CodeGraphRAG\.tmp\algorithm-graph.json")

# === 读取图数据 ===
with open(graph_path, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

relationships = graph_data.get("relationships", [])

# === 构建邻接表 ===
graph = {}
for r in relationships:
    from_id = r["from_id"]
    to_id = r["to_id"]
    graph.setdefault(from_id, []).append(to_id)


# === 检测有向图中的环 ===
def detect_cycles(graph):
    visited = set()
    stack = set()
    cycles = []

    def dfs(node, path):
        if node in stack:
            cycle_path = path[path.index(node):] + [node]
            cycles.append(cycle_path)
            return
        if node in visited:
            return

        visited.add(node)
        stack.add(node)
        for neighbor in graph.get(node, []):
            dfs(neighbor, path + [neighbor])
        stack.remove(node)

    for node in graph.keys():
        dfs(node, [node])

    return cycles


cycles = detect_cycles(graph)

# === 打印检测结果 ===
if cycles:
    print("⚠️ 检测到环:")
    for c in cycles:
        print(" -> ".join(map(str, c)))
else:
    print("✅ 没有检测到环。")
