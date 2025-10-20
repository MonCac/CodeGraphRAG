from loguru import logger
from pydantic_ai import Tool
from rich.console import Console

from ..graph_updater import MemgraphIngestor
from ..services.llm import CypherGenerator, LLMGenerationError


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

    async def query_codebase_knowledge_graph(batch_queries: dict) -> list[dict]:
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
