# test_enre_loader.py
from codebase_rag.enre.loader import ENRELoader


def test_enre_loader(repo_path: str):
    print("=== Running ENRELoader Test ===")
    loader = ENRELoader(repo_path)

    # Step 1: 运行 ENRE 分析
    loader.run_enre_analysis()

    # Step 2: 获取 nodes 和 relationships
    nodes, relationships = loader.get_nodes_and_relationships()

    # 打印一些结果
    print(f"Total nodes: {len(nodes)}")
    print(f"Total relationships: {len(relationships)}")

    # 打印前几个节点和关系示例
    print("\nSample nodes:")
    for node in nodes[:5]:
        print(node)

    print("\nSample relationships:")
    for rel in relationships[:5]:
        print(rel)

    loader.save_json()

if __name__ == "__main__":
    # ⚠️ 修改为你要测试的 repo 路径
    repo_path = "/Users/moncheri/Downloads/main/PMD-main/pmd-java"
    test_enre_loader(repo_path)
