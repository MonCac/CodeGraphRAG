import json
#
# from codebase_rag.main import find_first_antipattern_json_in_parent_dir
# from codebase_rag.services.graph_file_classifier.classifier import GraphFileClassifier


def map_antipattern_node_to_project(antinode, project_nodes, exclude_keys={"id", "parentId", "external", "additionalBin", "File", "parameter", "rawType", "enhancement"}):
    """
    将单个 antipattern 节点映射到 project_nodes 中对应节点的 node_id。
    匹配规则：
        - labels 必须一致（顺序无关）
        - properties 去除 exclude_keys 后递归比较一致

    参数:
        antipattern_node: dict，包含 'labels' 和 'properties'
        project_nodes: list of dict，每个元素同样包含 'labels' 和 'properties'
        exclude_keys: set，要排除的 properties 字段，默认排除 id, parentId, external

    返回:
        匹配的 project_nodes 中节点的 node_id，找不到返回 None
    """

    def deep_filter_properties(props):
        """递归过滤 dict，去除 exclude_keys，支持嵌套 dict 和 list。"""
        if isinstance(props, dict):
            return {
                k: deep_filter_properties(v)
                for k, v in props.items()
                if k not in exclude_keys
            }
        elif isinstance(props, list):
            return [deep_filter_properties(item) for item in props]
        else:
            return props

    def deep_compare(a, b):
        """递归比较两个结构是否相等。"""
        if type(a) != type(b):
            return False
        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            return all(deep_compare(a[k], b[k]) for k in a)
        elif isinstance(a, list):
            if len(a) != len(b):
                return False
            return all(deep_compare(x, y) for x, y in zip(a, b))
        else:
            return a == b

    alabels = tuple(sorted(antinode.get("labels", [])))
    aprops = antinode.get("properties", {}) or {}
    filtered_aprops = deep_filter_properties(aprops)

    for pnode in project_nodes:
        plabels = tuple(sorted(pnode.get("labels", [])))
        if alabels != plabels:
            continue
        pprops = pnode.get("properties", {}) or {}
        filtered_pprops = deep_filter_properties(pprops)

        if deep_compare(filtered_aprops, filtered_pprops):
            return pnode["node_id"]

    return None


def test_map_antipattern_nodes_to_project(antipattern_nodes_path, project_nodes_path):
    # 读取 antipattern 节点列表
    with open(antipattern_nodes_path, 'r', encoding='utf-8') as f:
        antipattern_graph = json.load(f)
    antipattern_nodes = antipattern_graph.get("nodes", [])

    # 读取 project 节点列表
    with open(project_nodes_path, 'r', encoding='utf-8') as f:
        project_graph = json.load(f)
    project_nodes = project_graph.get("nodes", [])

    for i, antipattern_node in enumerate(antipattern_nodes):
        matched_node_id = map_antipattern_node_to_project(antipattern_node, project_nodes)
        print(f"Antipattern node #{i} 匹配到的 project 节点 node_id: {matched_node_id}")


# 调用示例（替换为你真实的文件路径）
# test_map_antipattern_nodes_to_project('/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/tmp/20-graph.json', '/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/tmp/kafka-graph.json')


if __name__ == "__main__":
    from openai import OpenAI

    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        # sk-xxx替换为自己的key
        api_key='sk-9t1798hIfnUm1WyZ799fE44265Dc428696038561D341C516'
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(completion.choices[0].message)
    # graph_data = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/tmp/awd-final-result.json"
    # antipattern_to_update = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/test-project/AWD/86/before"
    # antipattern_type = "awd"
    # with open(graph_data, "r", encoding="utf-8") as f:
    #     graph_data = json.load(f)
    # classifier = GraphFileClassifier(graph_data, antipattern_type)
    # antipattern_json_path = find_first_antipattern_json_in_parent_dir(antipattern_to_update)
    # result1 = classifier.classify(antipattern_json_path)
    # print(result1)
