from typing import Dict, Set
from .base_strategy import BaseAntipatternStrategy
import os

from .utils import find_node_by_entity_and_location, find_related_files_by_relationships


class CHStrategy(BaseAntipatternStrategy):

    def find_direct_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        files = set()
        for item in antipattern_json.get("files", []):
            files.add(item)
        return files

    def find_indirect_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        child_method = antipattern_json["codeSnippets"][0]["childMethod"]

        child_node = find_node_by_entity_and_location(graph_data, child_method["entity"], child_method["location"])
        child_node_id = child_node.get("properties", {}).get("id")

        node_ids = set(filter(None, [child_node_id]))
        indirect = find_related_files_by_relationships(graph_data, node_ids)

        direct_related = self.find_direct_related_files(graph_data, antipattern_json)
        return indirect - direct_related
