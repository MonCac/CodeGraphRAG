from pathlib import Path

from loguru import logger
from pydantic_ai import RunContext, Tool

from ..graph_updater import MemgraphIngestor
from ..schemas import CodeSnippet


class CodeRetriever:
    """Service to retrieve code snippets using the graph and filesystem."""

    def __init__(self, project_root: str, ingestor: MemgraphIngestor):
        self.project_root = Path(project_root).resolve()
        self.ingestor = ingestor
        logger.info(f"CodeRetriever initialized with root: {self.project_root}")

    async def find_code_snippet(self, qualified_name: str) -> CodeSnippet:
        """Finds a code snippet by querying the graph for its location, handling all labels efficiently."""
        logger.info(f"[CodeRetriever] Searching for: {qualified_name}")

        query = """
            MATCH (n)
            WHERE n.qualifiedName = $qn
            OPTIONAL MATCH (p) WHERE (n.location IS NULL OR n.location.startLine IS NULL OR n.location.endLine IS NULL) 
            AND p.id = n.parentId
            RETURN 
                labels(n) AS labels,
                n.name AS name,
                coalesce(n.location.startLine, p.location.startLine) AS start,
                coalesce(n.location.endLine, p.location.endLine) AS end,
                coalesce(n.File, p.File) AS path,
                n.parentId AS parentId
            LIMIT 1
        """
        params = {"qn": qualified_name}

        try:
            results = self.ingestor.fetch_all(query, params)
            if not results:
                return CodeSnippet(
                    qualified_name=qualified_name,
                    source_code="",
                    file_path="",
                    line_start=0,
                    line_end=0,
                    found=False,
                    error_message="Entity not found in graph.",
                )

            res = results[0]
            labels = res.get("labels", [])
            file_path_str = res.get("path")
            start_line = res.get("start")
            end_line = res.get("end")

            # 特殊处理
            if "Package" in labels:
                return CodeSnippet(
                    qualified_name=qualified_name,
                    source_code="",
                    file_path="",
                    line_start=0,
                    line_end=0,
                    found=False,
                    error_message="Provided node is a Package, cannot extract code snippet.",
                )

            if "File" in labels:
                # 返回整个文件
                full_path = self.project_root / file_path_str
                try:
                    with full_path.open("r", encoding="utf-8") as f:
                        source_code = f.read()
                    return CodeSnippet(
                        qualified_name=qualified_name,
                        source_code=source_code,
                        file_path=file_path_str,
                        line_start=1,
                        line_end=source_code.count("\n") + 1,
                        docstring=None,
                    )
                except FileNotFoundError:
                    return CodeSnippet(
                        qualified_name=qualified_name,
                        source_code="",
                        file_path=file_path_str,
                        line_start=0,
                        line_end=0,
                        found=False,
                        error_message=f"File not found: {full_path}",
                    )

            if not all([file_path_str, start_line, end_line]):
                return CodeSnippet(
                    qualified_name=qualified_name,
                    source_code="",
                    file_path=file_path_str or "",
                    line_start=start_line or 0,
                    line_end=end_line or 0,
                    found=False,
                    error_message="Graph entry is missing location data.",
                )

            # 读取代码片段
            full_path = self.project_root / file_path_str
            try:
                with full_path.open("r", encoding="utf-8") as f:
                    all_lines = f.readlines()
                snippet_lines = all_lines[start_line - 1: end_line]
                source_code = "".join(snippet_lines)
            except FileNotFoundError:
                return CodeSnippet(
                    qualified_name=qualified_name,
                    source_code="",
                    file_path=file_path_str,
                    line_start=start_line,
                    line_end=end_line,
                    found=False,
                    error_message=f"File not found: {full_path}",
                )

            return CodeSnippet(
                qualified_name=qualified_name,
                source_code=source_code,
                file_path=file_path_str,
                line_start=start_line,
                line_end=end_line,
                docstring=None,
            )

        except Exception as e:
            logger.error(f"[CodeRetriever] Error: {e}", exc_info=True)
            return CodeSnippet(
                qualified_name=qualified_name,
                source_code="",
                file_path="",
                line_start=0,
                line_end=0,
                found=False,
                error_message=str(e),
            )


def create_code_retrieval_tool(code_retriever: CodeRetriever) -> Tool:
    """Factory function to create the code snippet retrieval tool."""

    async def get_code_snippet(ctx: RunContext, qualified_name: str) -> CodeSnippet:
        """Retrieves the source code for a given qualified name."""
        logger.info(f"[Tool:GetCode] Retrieving code for: {qualified_name}")
        return await code_retriever.find_code_snippet(qualified_name)

    return Tool(
        function=get_code_snippet,
        description="Retrieves the source code for a specific function, class, or method using its full qualified name.",
    )
