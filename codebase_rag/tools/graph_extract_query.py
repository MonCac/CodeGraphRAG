import json
import re
from typing import List, Any, Dict
from pydantic_ai import Agent
from loguru import logger
from pydantic_ai import Tool
from rich.console import Console
import ast

from ..graph_updater import MemgraphIngestor
from ..prompts import CYPHER_SYSTEM_PROMPT
from ..services.llm import CypherGenerator, LLMGenerationError, create_graph_extract_query_model


def create_graph_extract_query_tool(
        ingestor: MemgraphIngestor,
        cypher_gen: CypherGenerator,
        console: Console | None = None,
) -> Tool:
    """
    Tool that executes batch natural language queries for entities and dependencies
    based on JSON input. Returns raw query results.
    """
    if console is None:
        console = Console(width=None, force_terminal=True)

    async def query_codebase_knowledge_graph(batch_queries: Any) -> list[list | list[Any]]:
        """
        batch_queries example:
        {
            "queries": [
                {
                    "type": "entity",
                    "description": "Extract all nodes in the 'auth' module, including classes, methods, and variables, with their id, labels, and properties."
                },
                {
                    "type": "relationship",
                    "description": "Extract all dependencies between nodes in the 'auth' module, including method calls, variable definitions, and class inheritance."
                }
            ]
        }

        Returns a list of query results in the same order.
        """
        results_list = []
        print("-------------------------------------entertools---------------------------------------")
        queries = batch_queries.get("queries", [])
        if not queries:
            logger.warning("[Tool:QueryGraph] No queries found in batch input.")
            return results_list

        for q in queries:
            query_type = q.get("type")
            description = q.get("description")
            if not description or not query_type:
                logger.warning("[Tool:QueryGraph] Skipping query with missing type or description.")
                continue

            llm_input = f"Query type: {query_type}\nInstruction: {description}"

            try:
                logger.info(f"[Tool:QueryGraph] Generating Cypher for type '{query_type}': {description}")
                cypher_query = await cypher_gen.generate(llm_input)
                logger.info(f"[Tool:QueryGraph] Generated Cypher:\n{cypher_query}")

                query_results = ingestor.fetch_all(cypher_query)
                results_list.append(query_results)

            except LLMGenerationError as e:
                logger.error(f"[Tool:QueryGraph] LLM generation error: {e}")
                results_list.append([])

            except Exception as e:
                logger.error(f"[Tool:QueryGraph] Query execution error: {e}", exc_info=True)
                results_list.append([])

        return results_list

    return Tool(
        function=query_codebase_knowledge_graph,
        description=(
            "Execute batch natural language queries against the codebase knowledge graph. "
            "Input JSON should contain a 'queries' list with 'type' ('entity' or 'relationship') "
            "and 'description' (natural language query instruction). Returns raw query results for each entry."
        ),
    )


async def query_codebase_with_agent_batch(
        user_input: str,
        ingestor,
        console: Console | None = None,
) -> list[Dict]:
    """
    批量执行自然语言图谱查询，支持动态替换 agent 的 system_prompt。

    Args:
        agent: pydantic_ai.Agent 实例
        user_input: JSON 字符串，包含 'queries' 列表，每条 query 有 'type' 和 'description'
        ingestor: 执行 Cypher 查询的对象
        console: 可选，用于打印日志

    Returns:
        查询结果列表，每个 query 对应一个列表
    """
    agent = create_graph_extract_query_model()
    if console is None:
        console = Console(width=None, force_terminal=True)

    # 动态替换 system_prompt（如果提供）
    console.print(f"[bold blue]Agent system_prompt before.:{agent.system_prompt}[/bold blue]")
    agent.system_prompt = CYPHER_SYSTEM_PROMPT
    console.print(f"[bold blue]Agent system_prompt replaced.:{agent.system_prompt}[/bold blue]")

    results_list = []

    # 解析 JSON
    try:
        batch_queries = robust_parse_json(user_input)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse user_input as JSON: {e}")
        console.print(f"[red]Failed to parse JSON input:[/red] {e}")
        return results_list

    queries = batch_queries.get("queries", [])
    if not queries:
        logger.warning("No queries found in input.")
        return results_list

    for idx, q in enumerate(queries, start=1):
        query_type = q.get("type")
        description = q.get("description")
        if not query_type or not description:
            logger.warning(f"Skipping query #{idx} with missing type or description.")
            results_list.append([])
            continue

        llm_input = f"Query type: {query_type}\nInstruction: {description}"
        console.print(f"[cyan]Processing query #{idx}:[/cyan] {description}")

        try:
            # 用 agent 生成 Cypher
            response = await agent.run(llm_input)
            cypher_query = getattr(response, "output", str(response)).strip()
            console.print(f"[magenta]Generated Cypher:[/magenta] {cypher_query}")

            # 执行查询
            query_results = ingestor.fetch_all(cypher_query)
            results_list.append(query_results)
            console.print(f"[green]Query #{idx} done, {len(query_results)} results.[/green]")

        except Exception as e:
            logger.error(f"Error processing query #{idx}: {e}", exc_info=True)
            console.print(f"[red]Error processing query #{idx}:[/red] {e}")
            results_list.append([])

    # 缺少写入数据库的操作
    # 需要存储数据、删除数据库中原有内容，更新数据库。
    # 要用到 graph_service.py 和 graph_updater.py 中的内容

    return results_list


def robust_parse_json(user_input: str):
    """
    安全解析 JSON：
    - 去掉开头/结尾多余的括号
    - 单引号转双引号
    - 移除多余逗号
    """
    text = user_input.strip()

    # 去掉开头多余 '{'
    while text.startswith('{{'):
        text = text[1:]

    # 去掉末尾多余 '}'
    while text.endswith('}}'):
        text = text[:-1]

    # 替换 Python 风格单引号为 JSON 双引号
    text = re.sub(r"(?<!\")'([^']*?)'(?!\")", r'"\1"', text)

    # 移除列表或对象末尾多余逗号
    text = re.sub(r",(\s*[\]}])", r"\1", text)

    # 尝试解析
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse input as JSON: {e}")