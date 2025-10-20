"""Type inference engine for determining variable types."""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from tree_sitter import Node

from .import_processor import ImportProcessor
from .java_type_inference import JavaTypeInferenceEngine

if TYPE_CHECKING:
    from .factory import ASTCacheProtocol


class TypeInferenceEngine:
    """Handles type inference for local variables and method returns."""

    def __init__(
            self,
            import_processor: ImportProcessor,
            function_registry: Any,
            repo_path: Path,
            project_name: str,
            ast_cache: "ASTCacheProtocol",
            queries: dict[str, Any],
            module_qn_to_file_path: dict[str, Path],
            class_inheritance: dict[str, list[str]],
    ):
        self.import_processor = import_processor
        self.function_registry = function_registry
        self.repo_path = repo_path
        self.project_name = project_name
        self.ast_cache = ast_cache
        self.queries = queries
        self.module_qn_to_file_path = module_qn_to_file_path
        self.class_inheritance = class_inheritance

        # Java-specific type inference engine (lazy-loaded)
        self._java_type_inference: JavaTypeInferenceEngine | None = None

    @property
    def java_type_inference(self) -> JavaTypeInferenceEngine:
        """Lazy-loaded Java type inference engine."""
        if self._java_type_inference is None:
            self._java_type_inference = JavaTypeInferenceEngine(
                import_processor=self.import_processor,
                function_registry=self.function_registry,
                repo_path=self.repo_path,
                project_name=self.project_name,
                ast_cache=self.ast_cache,
                queries=self.queries,
                module_qn_to_file_path=self.module_qn_to_file_path,
                class_inheritance=self.class_inheritance,
            )
        return self._java_type_inference

    def build_local_variable_type_map(
            self, caller_node: Node, module_qn: str, language: str
    ) -> dict[str, str]:
        """
        Build a map of local variable names to their inferred types within a function.
        This enables resolution of instance method calls like user.get_name().
        """
        local_var_types: dict[str, str] = {}

        if language == "java":
            # Use Java-specific type inference
            return self._build_java_local_variable_type_map(caller_node, module_qn)
        else:
            # Unsupported language
            return local_var_types

    def _build_java_local_variable_type_map(
            self, caller_node: Node, module_qn: str
    ) -> dict[str, str]:
        """Build local variable type map for Java using JavaTypeInferenceEngine."""
        return self.java_type_inference.build_java_variable_type_map(
            caller_node, module_qn
        )
