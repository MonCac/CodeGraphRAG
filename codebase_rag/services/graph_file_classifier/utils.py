import re
from typing import Dict, Set, Optional


def extract_all_files_from_graph(graph_data: Dict) -> Set[str]:
    """
    从 graph_data 的节点中提取所有文件路径（来自 properties 中的 'File' 字段）。
    返回一个去重的文件路径集合。
    """
    files = set()

    nodes = graph_data.get("nodes", [])
    for node in nodes:
        properties = node.get("properties", {})
        file_path = properties.get("File")
        if file_path:
            files.add(file_path)

    return files


def parse_location_str(location_str: str) -> Optional[tuple[int, int]]:
    """
    解析类似 "107–109" 的字符串，返回 (startLine, endLine)
    注意：中间破折号可能是不同编码的长短横杠，做兼容处理。
    """
    if not location_str:
        return None
    # 用正则分割破折号（–、-）
    parts = re.split(r"[–-]", location_str)
    if len(parts) != 2:
        return None
    try:
        start_line = int(parts[0].strip())
        end_line = int(parts[1].strip())
        return start_line, end_line
    except ValueError:
        return None


def location_matches(node_location: Dict, target_start: int, target_end: int) -> bool:
    """
    判断 graph_data 节点中的 location 是否包含目标行范围
    必须 node_location 的 startLine == target_start 且 endLine == target_end 即认为匹配
    """
    if not node_location:
        return False
    start_line = node_location.get("startLine")
    end_line = node_location.get("endLine")
    if start_line is None or end_line is None:
        return False
    return start_line == target_start and end_line == target_end


def find_node_by_entity_and_location(graph_data: Dict, entity: str, location_str: str) -> Optional[Dict]:
    """
    在 graph_data['nodes'] 中查找满足：
    - properties.qualifiedName == entity
    - location.startLine 和 endLine 覆盖 location_str 中的行范围

    返回匹配到的节点（字典），找不到返回 None
    """
    location_range = parse_location_str(location_str)
    if location_range is None:
        return None
    target_start, target_end = location_range

    for node in graph_data.get("nodes", []):
        properties = node.get("properties", {})
        qualified_name = properties.get("qualifiedName")
        if qualified_name != entity:
            continue
        node_location = properties.get("location")
        if location_matches(node_location, target_start, target_end):
            return node
    return None


def find_related_files_by_relationships(graph_data: Dict, node_ids: Set[int]) -> Set[str]:
    """
    根据节点ID集合，遍历 relationships 找相关节点的文件路径。
    只要相关节点有 'File' 属性，则加入结果集合。
    """
    related_files = set()
    relationships = graph_data.get("relationships", [])
    nodes = graph_data.get("nodes", [])

    # 建立 id -> node 映射，加速查找
    id_to_node = {node.get("properties", {}).get("id"): node for node in nodes if
                  node.get("properties", {}).get("id") is not None}

    for rel in relationships:
        from_id = rel.get("from_id")
        to_id = rel.get("to_id")

        # 判断是否包含 node_ids 中的 id
        if from_id in node_ids:
            other_id = to_id
        elif to_id in node_ids:
            other_id = from_id
        else:
            continue

        # 找到对应的节点
        other_node = id_to_node.get(other_id)
        if not other_node:
            continue

        # 取文件路径
        file_path = other_node.get("properties", {}).get("File")
        if file_path:
            related_files.add(file_path)

    return related_files


def extract_all_child_methods(antipattern_json):
    results = []

    def dfs(obj):
        if isinstance(obj, dict):
            if "childMethod" in obj and isinstance(obj["childMethod"], dict):
                results.append(obj["childMethod"])
            for v in obj.values():
                dfs(v)
        elif isinstance(obj, list):
            for item in obj:
                dfs(item)

    dfs(antipattern_json)
    return results
