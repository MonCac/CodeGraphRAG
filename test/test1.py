# test_enre_loader.py
import asyncio

from codebase_rag.enre.loader import ENRELoader
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models import ModelMessage
from pydantic_ai.models import ModelRequestParameters

from codebase_rag.services.llm import create_semantic_model


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


def main():
    agent = create_semantic_model()

    # 调用模型
    response = agent.run_sync("Hello World")
    print(type(response))
    print(dir(response))


    # 输出生成文本
    print(response.output)  # 或 response.content

if __name__ == "__main__":
    #⚠️ 修改为你要测试的 repo 路径
    # repo_path = "/Users/moncheri/Downloads/main/PMD-main/pmd-java"
    # test_enre_loader(repo_path)
    # help(OpenAIChatModel)
    main()
