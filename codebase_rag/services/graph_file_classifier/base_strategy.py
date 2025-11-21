from typing import Dict, Set


class BaseAntipatternStrategy:
    """
    所有 antipattern type 的父类。
    每种类型必须实现不同的 direct / indirect 提取方法。
    """

    def find_direct_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        raise NotImplementedError

    def find_indirect_related_files(self, graph_data: Dict, antipattern_json: Dict) -> Set[str]:
        raise NotImplementedError
