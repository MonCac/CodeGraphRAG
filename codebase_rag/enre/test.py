from pathlib import Path
import json
from codebase_rag.enre.enre_graph_analyzer import ENREGraphAnalyzer
from codebase_rag.config import settings

if __name__ == "__main__":
    target_repo_path = settings.TARGET_REPO_PATH
    antipattern_relation_path = settings.ANTIPATTERN_RELATION_PATH

    enre_graph_analyzer = ENREGraphAnalyzer(project_path=target_repo_path, antipattern_path=antipattern_relation_path)

    enre_graph_analyzer.save_subgraph(
        output_path=".tmp/my_project_subgraph.json",
        max_depth=3,
        max_nodes=100000
    )
