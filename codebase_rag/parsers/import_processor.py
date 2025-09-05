"""Import processor for parsing and resolving import statements."""

from pathlib import Path
from typing import Any

from loguru import logger
from tree_sitter import Node

from ..language_config import LanguageConfig
from .utils import get_query_cursor, safe_decode_text, safe_decode_with_fallback


class ImportProcessor:
    """Handles parsing and processing of import statements."""

    def __init__(
            self,
            repo_path_getter: Any,
            project_name_getter: Any,
            ingestor: Any | None = None,
    ) -> None:
        self._repo_path_getter = repo_path_getter
        self._project_name_getter = project_name_getter
        self.ingestor = ingestor
        self.import_mapping: dict[str, dict[str, str]] = {}

    @property
    def repo_path(self) -> Path:
        """Get the current repo path dynamically."""
        if callable(self._repo_path_getter):
            result = self._repo_path_getter()
            return result if isinstance(result, Path) else Path(result)
        return (
            Path(self._repo_path_getter)
            if isinstance(self._repo_path_getter, str)
            else self._repo_path_getter
        )

    @property
    def project_name(self) -> str:
        """Get the current project name dynamically."""
        if callable(self._project_name_getter):
            result = self._project_name_getter()
            return str(result)
        return str(self._project_name_getter)

    def parse_imports(
            self, root_node: Node, module_qn: str, language: str, queries: dict[str, Any]
    ) -> None:
        """Parse import statements and build import mapping for the module."""
        if language not in queries or not queries[language].get("imports"):
            return

        lang_config = queries[language]["config"]
        imports_query = queries[language]["imports"]

        self.import_mapping[module_qn] = {}

        try:
            cursor = get_query_cursor(imports_query)
            captures = cursor.captures(root_node)

            # Handle different language import patterns
            if language == "java":
                self._parse_java_imports(captures, module_qn)
            else:
                # Generic fallback for other languages
                self._parse_generic_imports(captures, module_qn, lang_config)

            logger.debug(
                f"Parsed {len(self.import_mapping[module_qn])} imports in {module_qn}"
            )

            # Create IMPORTS relationships for each parsed import
            if self.ingestor and module_qn in self.import_mapping:
                for local_name, full_name in self.import_mapping[module_qn].items():
                    self.ingestor.ensure_relationship_batch(
                        ("Module", "qualified_name", module_qn),
                        "IMPORTS",
                        ("Module", "qualified_name", full_name),
                    )
                    logger.debug(
                        f"  Created IMPORTS relationship: {module_qn} -> {full_name}"
                    )

        except Exception as e:
            logger.warning(f"Failed to parse imports in {module_qn}: {e}")

    def _resolve_relative_import(self, relative_node: Node, module_qn: str) -> str:
        """Resolve relative imports like '.module' or '..parent.module'."""
        module_parts = module_qn.split(".")[1:]  # Remove project name

        # Count the dots to determine how many levels to go up
        dots = 0
        module_name = ""

        for child in relative_node.children:
            if child.type == "import_prefix":
                decoded_text = safe_decode_text(child)
                if not decoded_text:
                    continue
                dots = len(decoded_text)
            elif child.type == "dotted_name":
                decoded_name = safe_decode_text(child)
                if not decoded_name:
                    continue
                module_name = decoded_name

        # Calculate the target module - dots corresponds to levels to go up
        target_parts = module_parts[:-dots] if dots > 0 else module_parts

        if module_name:
            target_parts.extend(module_name.split("."))

        return ".".join(target_parts)

    def _parse_java_imports(self, captures: dict, module_qn: str) -> None:
        """Parse Java import statements."""

        for import_node in captures.get("import", []):
            if import_node.type == "import_declaration":
                is_static = False
                imported_path = None
                is_wildcard = False

                # Parse import declaration
                for child in import_node.children:
                    if child.type == "static":
                        is_static = True
                    elif child.type == "scoped_identifier":
                        imported_path = safe_decode_with_fallback(child)
                    elif child.type == "asterisk":
                        is_wildcard = True

                if not imported_path:
                    continue

                if is_wildcard:
                    # import java.util.*; - wildcard import
                    logger.debug(f"Java wildcard import: {imported_path}.*")
                    # Store wildcard import for potential future use
                    self.import_mapping[module_qn][f"*{imported_path}"] = imported_path
                else:
                    # import java.util.List; or import static java.lang.Math.PI;
                    parts = imported_path.split(".")
                    if parts:
                        imported_name = parts[-1]  # Last part is class/method name
                        if is_static:
                            # Static import - method/field can be used directly
                            self.import_mapping[module_qn][imported_name] = (
                                imported_path
                            )
                            logger.debug(
                                f"Java static import: {imported_name} -> "
                                f"{imported_path}"
                            )
                        else:
                            # Regular class import
                            self.import_mapping[module_qn][imported_name] = (
                                imported_path
                            )
                            logger.debug(
                                f"Java import: {imported_name} -> {imported_path}"
                            )

    def _parse_generic_imports(
            self, captures: dict, module_qn: str, lang_config: LanguageConfig
    ) -> None:
        """Generic fallback import parsing for other languages."""

        for import_node in captures.get("import", []):
            logger.debug(
                f"Generic import parsing for {lang_config.name}: {import_node.type}"
            )
