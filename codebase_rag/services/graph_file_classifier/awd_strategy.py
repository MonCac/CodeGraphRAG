from typing import Dict, Set
from .base_strategy import BaseAntipatternStrategy
from .utils import find_node_by_entity_and_location, find_related_files_by_relationships, extract_all_child_methods


class AWDStrategy(BaseAntipatternStrategy):

    def find_direct_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        files = set()
        for item in antipattern_json.get("files", []):
            files.add(item)
        return files

    def find_indirect_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        child_methods = extract_all_child_methods(antipattern_json)
        all_indirect = set()
        all_direct = set()
        for child_method in child_methods:
            child_node = find_node_by_entity_and_location(
                graph_data,
                child_method["entity"],
                child_method["location"]
            )
            if not child_node:
                continue
            child_node_id = child_node.get("properties", {}).get("id")
            if not child_node_id:
                continue
            indirect = find_related_files_by_relationships(
                graph_data,
                node_ids={child_node_id}
            )
            all_indirect |= indirect
        all_direct = self.find_direct_related_files(graph_data, antipattern_json)
        return all_indirect - all_direct
