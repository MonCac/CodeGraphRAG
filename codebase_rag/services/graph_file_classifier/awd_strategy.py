from typing import Dict, Set
from .base_strategy import BaseAntipatternStrategy


class AWDStrategy(BaseAntipatternStrategy):

    def find_direct_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        # TODO: C 类型的直接关联逻辑
        return set()

    def find_indirect_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        # TODO: C 类型的间接关联逻辑
        return set()
