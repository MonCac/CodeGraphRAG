import json
import os


def analyze_antipattern_json_normal(repo_out_path, antipattern_path, output_path):
    # ---------- 1. 读取两个输入文件 ----------
    with open(repo_out_path, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)

    with open(antipattern_path, 'r', encoding='utf-8') as f:
        anti_data = json.load(f)

    # ---------- 2. 从antipattern读取 dependencyChain ----------
    dep_chain = anti_data.get("dependencyChain", [])
    if not dep_chain:
        print("未在 antipattern.json 中找到 dependencyChain")
        return

    from_entity = dep_chain[0].get("from")
    to_entity = dep_chain[0].get("to")

    if not from_entity or not to_entity:
        print("dependencyChain 中缺少 from/to 信息")
        return

    # ---------- 3. 从repo_out中找到对应的id ----------
    variables = repo_data.get("variables", [])
    id_map = {}  # qualifiedName -> id

    for var in variables:
        qname = var.get("qualifiedName")
        if qname in [from_entity, to_entity]:
            id_map[qname] = var.get("id")

    if len(id_map) < 2:
        print("在 repo_out.json 中未找到匹配的 from/to 变量。")
        return

    from_id = id_map.get(from_entity)
    to_id = id_map.get(to_entity)

    # ---------- 4. 找出与这两个id相关的cells ----------
    cells = repo_data.get("cells", [])
    target_ids = {from_id, to_id}
    related_ids = set(target_ids)  # 初始包含from_id和to_id
    related_cells = []

    for cell in cells:
        src = cell.get("src")
        dest = cell.get("dest")

        # 如果src或dest中包含目标id之一，则关联
        if src in target_ids or dest in target_ids:
            related_cells.append(cell)

            # 把另一端的id也加入related_ids
            if src in target_ids:
                related_ids.add(dest)
            if dest in target_ids:
                related_ids.add(src)

    # ---------- 5. 获取所有相关的variables ----------
    related_variables = [v for v in variables if v.get("id") in related_ids]

    # ---------- 6. 构造 relatedFiles ----------
    related_files = set()
    for var in related_variables:
        file_name = var.get("File")
        additional_bin = var.get("additionalBin")
        if file_name and additional_bin:
            bin_path = additional_bin.get("binPath")
            if bin_path:
                file_path = os.path.join(bin_path, file_name)
                related_files.add(file_path)

    result = {
        "nodes": related_variables,
        "relationships": related_cells,
        "relatedFiles": list(related_files)
    }

    # ---------- 7. 写入结果 ----------
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"分析完成，结果已写入 {output_path}")


def analyze_antipattern_json_change_format(repo_out_path, antipattern_path, output_path):
    # ---------- 1. 读取两个输入文件 ----------
    with open(repo_out_path, 'r', encoding='utf-8') as f:
        repo_data = json.load(f)

    with open(antipattern_path, 'r', encoding='utf-8') as f:
        anti_data = json.load(f)

    # ---------- 2. 从antipattern读取 dependencyChain ----------
    dep_chain = anti_data.get("dependencyChain", [])
    if not dep_chain:
        print("未在 antipattern.json 中找到 dependencyChain")
        return

    from_entity = dep_chain[0].get("from")
    to_entity = dep_chain[0].get("to")

    if not from_entity or not to_entity:
        print("dependencyChain 中缺少 from/to 信息")
        return

    # ---------- 3. 从repo_out中找到对应的id ----------
    variables = repo_data.get("variables", [])
    id_map = {}  # qualifiedName -> id

    for var in variables:
        qname = var.get("qualifiedName")
        if qname in [from_entity, to_entity]:
            id_map[qname] = var.get("id")

    if len(id_map) < 2:
        print("在 repo_out.json 中未找到匹配的 from/to 变量。")
        return

    from_id = id_map.get(from_entity)
    to_id = id_map.get(to_entity)

    # ---------- 4. 找出与这两个id相关的cells ----------
    cells = repo_data.get("cells", [])
    target_ids = {from_id, to_id}
    related_ids = set(target_ids)  # 初始包含from_id和to_id
    related_cells = []

    for cell in cells:
        src = cell.get("src")
        dest = cell.get("dest")

        # 如果src或dest中包含目标id之一，则关联
        if src in target_ids or dest in target_ids:
            related_cells.append(cell)

            # 把另一端的id也加入related_ids
            if src in target_ids:
                related_ids.add(dest)
            if dest in target_ids:
                related_ids.add(src)

    # ---------- 5. 获取所有相关的variables并转换结构 ----------
    related_variables = []
    for v in variables:
        if v.get("id") in related_ids:
            # 创建properties对象，包含除id和category外的所有字段
            properties = v.copy()
            # 移除id和category字段，因为它们已经单独处理了
            properties.pop("id", None)
            properties.pop("category", None)

            # 转换variable结构
            transformed_var = {
                "node_id": v.get("id"),
                "labels": [v.get("category", "")],
                "properties": properties
            }
            related_variables.append(transformed_var)

    # ---------- 6. 转换cells结构 ----------
    transformed_cells = []
    for cell in related_cells:
        values = cell.get("values", {})

        # 确定type：找到values中值为1的key作为type
        cell_type = ""
        for key, value in values.items():
            if value == 1 and key != "loc" and key != "arguments":
                cell_type = key
                break

        # 构建转换后的cell
        transformed_cell = {
            "from_id": cell.get("src"),
            "to_id": cell.get("dest"),
            "type": cell_type,
            "properties": {
                "loc": values.get("loc", {})
            }
        }
        # 如果loc为空，移除该属性
        if not transformed_cell["properties"]["loc"]:
            transformed_cell["properties"].pop("loc")

        # 如果properties为空，设置为空对象
        if not transformed_cell["properties"]:
            transformed_cell["properties"] = {}

        transformed_cells.append(transformed_cell)

    # ---------- 7. 构造 relatedFiles ----------
    related_files = set()
    for var in related_variables:
        file_name = var["properties"].get("File")
        additional_bin = var["properties"].get("additionalBin")
        if file_name and additional_bin:
            bin_path = additional_bin.get("binPath")
            if bin_path:
                file_path = os.path.join(bin_path, file_name)
                related_files.add(file_path)

    result = {
        "nodes": related_variables,
        "relationships": transformed_cells,
        "relatedFiles": list(related_files)
    }

    # ---------- 7. 写入结果 ----------
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"分析完成，结果已写入 {output_path}")

if __name__ == "__main__":
    repo_out_path = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/.tmp/kafka-out.json"
    antipattern_path = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/test-project/test1/20/kafka_20_ch_antipattern.json"
    output_path = "/Users/moncheri/Downloads/main/重构/反模式修复数据集构建/CodeGraphRAG/.tmp/analyze_antipattern_change_format_output.json"
    analyze_antipattern_json_change_format(repo_out_path, antipattern_path, output_path)