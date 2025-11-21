import json
import os
from pathlib import Path
from typing import Dict, List

from .ch_strategy import CHStrategy
from .mh_strategy import MHStrategy
from .awd_strategy import AWDStrategy
from .utils import extract_all_files_from_graph

STRATEGIES = {
    "ch": CHStrategy,
    "mh": MHStrategy,
    "awd": AWDStrategy
}


class GraphFileClassifier:
    def __init__(self, graph_data: Dict, antipattern_type: str):
        self.graph_data = graph_data

        if antipattern_type not in STRATEGIES:
            raise ValueError(f"Unknown antipattern_type: {antipattern_type}")

        self.strategy = STRATEGIES[antipattern_type]()

    def classify(self, antipattern_json_path: Path | str) -> Dict[str, List[str]]:
        with open(antipattern_json_path, "r", encoding="utf-8") as f:
            antipattern_json = json.load(f)
        direct_files = self.strategy.find_direct_related_files(
            self.graph_data,
            antipattern_json
        )

        indirect_files = self.strategy.find_indirect_related_files(
            self.graph_data,
            antipattern_json
        )

        # 获取所有文件
        all_files = extract_all_files_from_graph(self.graph_data)

        other_files = all_files - direct_files - indirect_files

        return {
            "direct_related": sorted(list(direct_files)),
            "indirect_related": sorted(list(indirect_files)),
            "other_files": sorted(list(other_files))
        }


