from typing import Dict, Set
from .base_strategy import BaseAntipatternStrategy


class MHStrategy(BaseAntipatternStrategy):

    def find_direct_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        files = set()
        for item in antipattern_json.get("files", []):
            files.add(item)
        return files

    def find_indirect_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        # 认为 MH 的 indirect 为空
        return set()
