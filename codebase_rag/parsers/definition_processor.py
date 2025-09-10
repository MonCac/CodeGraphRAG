"""Definition processor for extracting functions, classes and methods."""

import json
import re
import textwrap
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any

import toml
from loguru import logger
from tree_sitter import Node, Query, QueryCursor

from ..language_config import LanguageConfig
from ..services.graph_service import MemgraphIngestor

# No longer need constants import - using Tree-sitter directly

from .import_processor import ImportProcessor
from .java_utils import extract_java_method_info
from .python_utils import resolve_class_name
from .utils import ingest_exported_function, ingest_method


class DefinitionProcessor:
    """Handles processing of function, class, and method definitions."""

    def __init__(
            self,
            ingestor: MemgraphIngestor,
            repo_path: Path,
            project_name: str,
            function_registry: Any,
            simple_name_lookup: dict[str, set[str]],
            import_processor: ImportProcessor,
            module_qn_to_file_path: dict[str, Path],
    ):
        self.ingestor = ingestor
        self.repo_path = repo_path
        self.project_name = project_name
        self.function_registry = function_registry
        self.simple_name_lookup = simple_name_lookup
        self.import_processor = import_processor
        self.module_qn_to_file_path = module_qn_to_file_path
        self.class_inheritance: dict[str, list[str]] = {}

    def _get_node_type_for_inheritance(self, qualified_name: str) -> str:
        """
        Determine the node type for inheritance relationships.
        Returns the type from the function registry, defaulting to "Class".
        """
        node_type = self.function_registry.get(qualified_name, "Class")
        return str(node_type)

    def _create_inheritance_relationship(
            self, child_node_type: str, child_qn: str, parent_qn: str
    ) -> None:
        """
        Create an INHERITS relationship between child and parent entities.
        Determines the parent type automatically from the function registry.
        """
        parent_type = self._get_node_type_for_inheritance(parent_qn)
        self.ingestor.ensure_relationship_batch(
            (child_node_type, "qualified_name", child_qn),
            "INHERITS",
            (parent_type, "qualified_name", parent_qn),
        )

    def process_file(
            self,
            file_path: Path,
            language: str,
            queries: dict[str, Any],
            structural_elements: dict[Path, str | None],
    ) -> tuple[Node, str] | None:
        """
        Parses a file, ingests its structure and definitions,
        and returns the AST for caching.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        relative_path = file_path.relative_to(self.repo_path)
        relative_path_str = str(relative_path)
        logger.info(f"Parsing and Caching AST for {language}: {relative_path_str}")

        try:
            # Check if language is supported
            if language not in queries:
                logger.warning(f"Unsupported language '{language}' for {file_path}")
                return None

            source_bytes = file_path.read_bytes()
            # We need access to parsers, but we'll get it through queries
            lang_queries = queries[language]
            parser = lang_queries.get("parser")
            if not parser:
                logger.warning(f"No parser available for {language}")
                return None

            tree = parser.parse(source_bytes)
            root_node = tree.root_node

            module_qn = ".".join(
                [self.project_name] + list(relative_path.with_suffix("").parts)
            )
            if file_path.name == "__init__.py":
                module_qn = ".".join(
                    [self.project_name] + list(relative_path.parent.parts)
                )
            elif file_path.name == "mod.rs":
                # In Rust, mod.rs represents the parent module directory
                module_qn = ".".join(
                    [self.project_name] + list(relative_path.parent.parts)
                )

            # Populate the module QN to file path mapping for efficient lookups
            self.module_qn_to_file_path[module_qn] = file_path

            self.ingestor.ensure_node_batch(
                "Module",
                {
                    "qualified_name": module_qn,
                    "name": file_path.name,
                    "path": relative_path_str,
                },
            )

            # Link Module to its parent Package/Folder
            parent_rel_path = relative_path.parent
            parent_container_qn = structural_elements.get(parent_rel_path)
            parent_label, parent_key, parent_val = (
                ("Package", "qualified_name", parent_container_qn)
                if parent_container_qn
                else (
                    ("Folder", "path", str(parent_rel_path))
                    if parent_rel_path != Path(".")
                    else ("Project", "name", self.project_name)
                )
            )
            self.ingestor.ensure_relationship_batch(
                (parent_label, parent_key, parent_val),
                "CONTAINS_MODULE",
                ("Module", "qualified_name", module_qn),
            )

            self.import_processor.parse_imports(root_node, module_qn, language, queries)
            self._ingest_missing_import_patterns(
                root_node, module_qn, language, queries
            )
            # Handle C++20 module-specific processing
            if language == "cpp":
                self._ingest_cpp_module_declarations(
                    root_node, module_qn, file_path, queries
                )
            self._ingest_all_functions(root_node, module_qn, language, queries)
            self._ingest_classes_and_methods(root_node, module_qn, language, queries)
            self._ingest_object_literal_methods(root_node, module_qn, language, queries)
            self._ingest_commonjs_exports(root_node, module_qn, language, queries)
            self._ingest_es6_exports(root_node, module_qn, language, queries)
            self._ingest_assignment_arrow_functions(
                root_node, module_qn, language, queries
            )
            self._ingest_prototype_inheritance(root_node, module_qn, language, queries)

            return root_node, language

        except Exception as e:
            logger.error(f"Failed to parse or ingest {file_path}: {e}")
            return None

    def process_dependencies(self, filepath: Path) -> None:
        """Parse various dependency files for external package dependencies."""
        file_name = filepath.name.lower()
        logger.info(f"  Parsing dependency file: {filepath}")

        try:
            if file_name == "pyproject.toml":
                self._parse_pyproject_toml(filepath)
            elif file_name == "requirements.txt":
                self._parse_requirements_txt(filepath)
            elif file_name == "package.json":
                self._parse_package_json(filepath)
            elif file_name == "cargo.toml":
                self._parse_cargo_toml(filepath)
            elif file_name == "go.mod":
                self._parse_go_mod(filepath)
            elif file_name == "gemfile":
                self._parse_gemfile(filepath)
            elif file_name == "composer.json":
                self._parse_composer_json(filepath)
            elif filepath.suffix.lower() == ".csproj":
                self._parse_csproj(filepath)
            else:
                logger.debug(f"    Unknown dependency file format: {filepath}")
        except Exception as e:
            logger.error(f"    Error parsing {filepath}: {e}")

    def _get_docstring(self, node: Node) -> str | None:
        """Extracts the docstring from a function or class node's body."""
        body_node = node.child_by_field_name("body")
        if not body_node or not body_node.children:
            return None
        first_statement = body_node.children[0]
        if (
                first_statement.type == "expression_statement"
                and first_statement.children[0].type == "string"
        ):
            text = first_statement.children[0].text
            if text is not None:
                result: str = text.decode("utf-8").strip("'\" \n")
                return result
        return None

    def _extract_decorators(self, node: Node) -> list[str]:
        """Extract decorator names from a decorated node."""
        decorators = []

        # Check if this node has a parent that is a decorated_definition
        current = node.parent
        while current:
            if current.type == "decorated_definition":
                # Get all decorator nodes
                for child in current.children:
                    if child.type == "decorator":
                        decorator_name = self._get_decorator_name(child)
                        if decorator_name:
                            decorators.append(decorator_name)
                break
            current = current.parent

        return decorators

    def _get_decorator_name(self, decorator_node: Node) -> str | None:
        """Extract the name from a decorator node (@decorator or @decorator(...))."""
        # Handle @decorator or @module.decorator
        for child in decorator_node.children:
            if child.type == "identifier":
                text = child.text
                if text is not None:
                    decorator_name: str = text.decode("utf8")
                    return decorator_name
            elif child.type == "attribute":
                # Handle @module.decorator
                text = child.text
                if text is not None:
                    attr_name: str = text.decode("utf8")
                    return attr_name
            elif child.type == "call":
                # Handle @decorator(...) - get the function being called
                func_node = child.child_by_field_name("function")
                if func_node:
                    if func_node.type == "identifier":
                        text = func_node.text
                        if text is not None:
                            func_name: str = text.decode("utf8")
                            return func_name
                    elif func_node.type == "attribute":
                        text = func_node.text
                        if text is not None:
                            func_attr_name: str = text.decode("utf8")
                            return func_attr_name
        return None

    def _extract_template_class_type(self, template_node: Node) -> str | None:
        """Extract the underlying class type from a template declaration."""
        # Look for the class/struct/union specifier within the template
        for child in template_node.children:
            if child.type == "class_specifier":
                return "Class"
            elif child.type == "struct_specifier":
                return "Class"
            elif child.type == "union_specifier":
                return "Union"
            elif child.type == "enum_specifier":
                return "Enum"
        return None

    def _extract_class_name(self, class_node: Node) -> str | None:
        """Extract class name, handling both class declarations and class expressions."""
        # For regular class declarations, try the name field first
        name_node = class_node.child_by_field_name("name")
        if name_node and name_node.text:
            return str(name_node.text.decode("utf8"))

        # For class expressions, look in parent variable_declarator
        # Pattern: const Animal = class { ... }
        current = class_node.parent
        while current:
            if current.type == "variable_declarator":
                # Find the identifier child (the name)
                for child in current.children:
                    if child.type == "identifier" and child.text:
                        return str(child.text.decode("utf8"))
            current = current.parent

        return None

    def _extract_function_name(self, func_node: Node) -> str | None:
        """Extract function name, handling both regular functions and arrow functions."""
        # For regular functions, try the name field first
        name_node = func_node.child_by_field_name("name")
        if name_node and name_node.text:
            return str(name_node.text.decode("utf8"))

        # For arrow functions, look in parent variable_declarator
        if func_node.type == "arrow_function":
            current = func_node.parent
            while current:
                if current.type == "variable_declarator":
                    # Find the identifier child (the name)
                    for child in current.children:
                        if child.type == "identifier" and child.text:
                            return str(child.text.decode("utf8"))
                current = current.parent

        return None

    def _generate_anonymous_function_name(self, func_node: Node, module_qn: str) -> str:
        """Generate a synthetic name for anonymous functions (IIFEs, callbacks, etc.)."""
        # Check if this is an IIFE pattern: function -> parenthesized_expression -> call_expression
        parent = func_node.parent
        if parent and parent.type == "parenthesized_expression":
            grandparent = parent.parent
            if grandparent and grandparent.type == "call_expression":
                # Check if the parenthesized expression is the function being called
                if grandparent.child_by_field_name("function") == parent:
                    func_type = (
                        "arrow" if func_node.type == "arrow_function" else "func"
                    )
                    return f"iife_{func_type}_{func_node.start_point[0]}_{func_node.start_point[1]}"

        # Check direct call pattern (less common but possible)
        if parent and parent.type == "call_expression":
            if parent.child_by_field_name("function") == func_node:
                return (
                    f"iife_direct_{func_node.start_point[0]}_{func_node.start_point[1]}"
                )

        # For other anonymous functions (callbacks, etc.), use location-based name
        return f"anonymous_{func_node.start_point[0]}_{func_node.start_point[1]}"

    def _ingest_all_functions(
            self, root_node: Node, module_qn: str, language: str, queries: dict[str, Any]
    ) -> None:
        """Extract and ingest all functions (including nested ones)."""
        lang_queries = queries[language]
        lang_config: LanguageConfig = lang_queries["config"]

        query = lang_queries["functions"]
        cursor = QueryCursor(query)
        captures = cursor.captures(root_node)

        func_nodes = captures.get("function", [])

        # Process regular functions
        for func_node in func_nodes:
            if not isinstance(func_node, Node):
                logger.warning(
                    f"Expected Node object but got {type(func_node)}: {func_node}"
                )
                continue
            if self._is_method(func_node, lang_config):
                continue

            is_exported = False  # Default for non-C++ languages
            # Extract function name - handle arrow functions specially
            func_name = self._extract_function_name(func_node)

            if not func_name:
                # Generate synthetic name for anonymous functions (IIFEs, callbacks, etc.)
                func_name = self._generate_anonymous_function_name(
                    func_node, module_qn
                )

            func_qn = (
                    self._build_nested_qualified_name(
                        func_node, module_qn, func_name, lang_config
                    )
                    or f"{module_qn}.{func_name}"
            )  # Fallback to simple name

            # Extract function properties
            decorators = self._extract_decorators(func_node)
            func_props: dict[str, Any] = {
                "qualified_name": func_qn,
                "name": func_name,
                "decorators": decorators,
                "start_line": func_node.start_point[0] + 1,
                "end_line": func_node.end_point[0] + 1,
                "docstring": self._get_docstring(func_node),
                "is_exported": is_exported,
            }
            logger.info(f"  Found Function: {func_name} (qn: {func_qn})")
            self.ingestor.ensure_node_batch("Function", func_props)

            self.function_registry[func_qn] = "Function"
            self.simple_name_lookup[func_name].add(func_qn)

            # Determine parent and create proper relationship
            parent_type, parent_qn = self._determine_function_parent(
                func_node, module_qn, lang_config
            )
            self.ingestor.ensure_relationship_batch(
                (parent_type, "qualified_name", parent_qn),
                "DEFINES",
                ("Function", "qualified_name", func_qn),
            )

    def _ingest_top_level_functions(
            self, root_node: Node, module_qn: str, language: str, queries: dict[str, Any]
    ) -> None:
        """Extract and ingest top-level functions. (Legacy method, replaced by _ingest_all_functions)"""
        # Keep for backward compatibility, but delegate to new method
        self._ingest_all_functions(root_node, module_qn, language, queries)

    def _build_nested_qualified_name(
            self,
            func_node: Node,
            module_qn: str,
            func_name: str,
            lang_config: LanguageConfig,
            skip_classes: bool = False,
    ) -> str | None:
        """Build qualified name for nested functions.

        Args:
            skip_classes: If True, skip class nodes in the path (used for object literal methods)
        """
        path_parts = []
        current = func_node.parent

        if not isinstance(current, Node):
            logger.warning(
                f"Unexpected parent type for node {func_node}: {type(current)}. "
                f"Skipping."
            )
            return None

        while current and current.type not in lang_config.module_node_types:
            # Handle functions (named and anonymous)
            if current.type in lang_config.function_node_types:
                if name_node := current.child_by_field_name("name"):
                    text = name_node.text
                    if text is not None:
                        path_parts.append(text.decode("utf8"))
                else:
                    # Check if this is an anonymous function that has been assigned a name
                    func_name_from_assignment = self._extract_function_name(current)
                    if func_name_from_assignment:
                        path_parts.append(func_name_from_assignment)
            # Handle classes
            elif current.type in lang_config.class_node_types:
                if skip_classes:
                    # For object literal methods, skip the class but continue up the tree
                    pass
                else:
                    # Check if we're inside a method that contains object literals
                    # If so, include the class name in the path. Otherwise, return None (this is a method)
                    if self._is_inside_method_with_object_literals(func_node):
                        if name_node := current.child_by_field_name("name"):
                            text = name_node.text
                            if text is not None:
                                path_parts.append(text.decode("utf8"))
                    else:
                        # Regular class method - return None
                        return None
            # Handle methods inside classes
            elif current.type == "method_definition":
                if name_node := current.child_by_field_name("name"):
                    text = name_node.text
                    if text is not None:
                        path_parts.append(text.decode("utf8"))

            current = current.parent

        path_parts.reverse()
        if path_parts:
            return f"{module_qn}.{'.'.join(path_parts)}.{func_name}"
        else:
            return f"{module_qn}.{func_name}"

    def _build_nested_qualified_name_for_class(
            self,
            class_node: Node,
            module_qn: str,
            class_name: str,
            lang_config: LanguageConfig,
    ) -> str | None:
        """Build qualified name for classes inside inline modules."""
        if not isinstance(class_node.parent, Node):
            return None
        return None

    def _is_method(self, func_node: Node, lang_config: LanguageConfig) -> bool:
        """Check if a function is actually a method inside a class."""
        current = func_node.parent
        if not isinstance(current, Node):
            return False

        while current and current.type not in lang_config.module_node_types:
            if current.type in lang_config.class_node_types:
                return True
            current = current.parent
        return False

    def _determine_function_parent(
            self, func_node: Node, module_qn: str, lang_config: LanguageConfig
    ) -> tuple[str, str]:
        """Determine the parent of a function (Module or another Function)."""
        current = func_node.parent
        if not isinstance(current, Node):
            return "Module", module_qn

        while current and current.type not in lang_config.module_node_types:
            if current.type in lang_config.function_node_types:
                if name_node := current.child_by_field_name("name"):
                    parent_text = name_node.text
                    if parent_text is None:
                        continue
                    parent_func_name = parent_text.decode("utf8")
                    if parent_func_qn := self._build_nested_qualified_name(
                            current, module_qn, parent_func_name, lang_config
                    ):
                        return "Function", parent_func_qn
                break

            current = current.parent

        return "Module", module_qn

    def _ingest_classes_and_methods(
            self, root_node: Node, module_qn: str, language: str, queries: dict[str, Any]
    ) -> None:
        """Extract and ingest classes and their methods."""
        lang_queries = queries[language]
        # Languages without classes (e.g., Lua) will not have a classes query
        if not lang_queries.get("classes"):
            return

        lang_config: LanguageConfig = lang_queries["config"]

        query = lang_queries["classes"]
        cursor = QueryCursor(query)
        captures = cursor.captures(root_node)
        class_nodes = captures.get("class", [])
        module_nodes = captures.get("module", [])


        for class_node in class_nodes:
            if not isinstance(class_node, Node):
                continue
            is_exported = False  # Default for non-C++ languages
            class_name = self._extract_class_name(class_node)
            if not class_name:
                continue
            # Build nested qualified name for classes inside inline modules
            nested_qn = self._build_nested_qualified_name_for_class(
                class_node, module_qn, class_name, lang_config
            )
            class_qn = nested_qn if nested_qn else f"{module_qn}.{class_name}"
            decorators = self._extract_decorators(class_node)
            class_props: dict[str, Any] = {
                "qualified_name": class_qn,
                "name": class_name,
                "decorators": decorators,
                "start_line": class_node.start_point[0] + 1,
                "end_line": class_node.end_point[0] + 1,
                "docstring": self._get_docstring(class_node),
                "is_exported": is_exported,
            }
            # Determine the correct node type based on the AST node type
            if class_node.type == "interface_declaration":
                node_type = "Interface"
                logger.info(f"  Found Interface: {class_name} (qn: {class_qn})")
            elif class_node.type in [
                "enum_declaration",
                "enum_specifier",
                "enum_class_specifier",
            ]:
                node_type = "Enum"
                logger.info(f"  Found Enum: {class_name} (qn: {class_qn})")
            elif class_node.type == "type_alias_declaration":
                node_type = "Type"
                logger.info(f"  Found Type: {class_name} (qn: {class_qn})")
            elif class_node.type == "struct_specifier":
                node_type = "Class"  # In C++, structs are essentially classes
                logger.info(f"  Found Struct: {class_name} (qn: {class_qn})")
            elif class_node.type == "union_specifier":
                node_type = "Union"
                logger.info(f"  Found Union: {class_name} (qn: {class_qn})")
            elif class_node.type == "template_declaration":
                # For template classes, check the actual class type within
                template_class = self._extract_template_class_type(class_node)
                node_type = template_class if template_class else "Class"
                logger.info(
                    f"  Found Template {node_type}: {class_name} (qn: {class_qn})"
                )
            elif class_node.type == "function_definition" and language == "cpp":
                # This is a misclassified exported class - determine type from text
                node_text = class_node.text.decode("utf-8") if class_node.text else ""
                if "export struct " in node_text:
                    node_type = "Class"  # In C++, structs are essentially classes
                    logger.info(
                        f"  Found Exported Struct: {class_name} (qn: {class_qn})"
                    )
                elif "export union " in node_text:
                    node_type = "Class"  # In C++, unions are also class-like
                    logger.info(
                        f"  Found Exported Union: {class_name} (qn: {class_qn})"
                    )
                elif "export template" in node_text:
                    node_type = "Class"  # Template class
                    logger.info(
                        f"  Found Exported Template Class: {class_name} (qn: {class_qn})"
                    )
                else:
                    node_type = "Class"  # Default to Class for exported classes
                    logger.info(
                        f"  Found Exported Class: {class_name} (qn: {class_qn})"
                    )
            else:
                node_type = "Class"
                logger.info(f"  Found Class: {class_name} (qn: {class_qn})")

            self.ingestor.ensure_node_batch(node_type, class_props)

            # Register the class/interface/enum itself in the function registry
            self.function_registry[class_qn] = node_type
            self.simple_name_lookup[class_name].add(class_qn)

            # Track inheritance
            parent_classes = self._extract_parent_classes(class_node, module_qn)
            self.class_inheritance[class_qn] = parent_classes

            self.ingestor.ensure_relationship_batch(
                ("Module", "qualified_name", module_qn),
                "DEFINES",
                (node_type, "qualified_name", class_qn),
            )

            # Create INHERITS relationships for each parent class
            for parent_class_qn in parent_classes:
                self._create_inheritance_relationship(
                    node_type, class_qn, parent_class_qn
                )

            # Handle Java interface implementations
            if class_node.type == "class_declaration":
                implemented_interfaces = self._extract_implemented_interfaces(
                    class_node, module_qn
                )
                for interface_qn in implemented_interfaces:
                    self._create_implements_relationship(
                        node_type, class_qn, interface_qn
                    )

            body_node = class_node.child_by_field_name("body")
            if not body_node:
                continue

            method_query = lang_queries["functions"]
            method_cursor = QueryCursor(method_query)
            method_captures = method_cursor.captures(body_node)
            method_nodes = method_captures.get("function", [])
            for method_node in method_nodes:
                if not isinstance(method_node, Node):
                    continue

                # Handle Java method overloading with parameter types
                method_qualified_name = None
                if language == "java":
                    method_info = extract_java_method_info(method_node)
                    method_name = method_info.get("name")
                    parameters = method_info.get("parameters", [])
                    if method_name:
                        if parameters:
                            # Create method signature with parameter types for overloading
                            param_signature = "(" + ",".join(parameters) + ")"
                            method_qualified_name = (
                                f"{class_qn}.{method_name}{param_signature}"
                            )
                        else:
                            # No parameters, use simple name
                            method_qualified_name = f"{class_qn}.{method_name}()"

                ingest_method(
                    method_node,
                    class_qn,
                    "Class",
                    self.ingestor,
                    self.function_registry,
                    self.simple_name_lookup,
                    self._get_docstring,
                    language,
                    self._extract_decorators,
                    method_qualified_name,
                )

                # Note: OVERRIDES relationships will be processed later after all methods are collected

        # Process inline modules (like Rust mod items)
        for module_node in module_nodes:
            if not isinstance(module_node, Node):
                continue

            module_name_node = module_node.child_by_field_name("name")
            if not module_name_node:
                continue
            text = module_name_node.text
            if text is None:
                continue
            module_name = text.decode("utf8")

            # Build nested qualified name for inline modules
            nested_qn = self._build_nested_qualified_name_for_class(
                module_node, module_qn, module_name, lang_config
            )
            inline_module_qn = nested_qn if nested_qn else f"{module_qn}.{module_name}"

            module_props: dict[str, Any] = {
                "qualified_name": inline_module_qn,
                "name": module_name,
                "path": f"inline_module_{module_name}",
            }
            logger.info(
                f"  Found Inline Module: {module_name} (qn: {inline_module_qn})"
            )
            self.ingestor.ensure_node_batch("Module", module_props)

    def process_all_method_overrides(self) -> None:
        """Process OVERRIDES relationships for all methods after collection is complete."""
        logger.info("--- Pass 4: Processing Method Override Relationships ---")

        # Process all methods to find overrides
        for method_qn in self.function_registry.keys():
            if self.function_registry[method_qn] == "Method":
                # Extract class_qn and method_name from method_qn
                if "." in method_qn:
                    parts = method_qn.rsplit(".", 1)
                    if len(parts) == 2:
                        class_qn, method_name = parts
                        self._check_method_overrides(method_qn, method_name, class_qn)

    def _check_method_overrides(
            self, method_qn: str, method_name: str, class_qn: str
    ) -> None:
        """Check if method overrides parent class methods using BFS traversal."""
        if class_qn not in self.class_inheritance:
            return

        # Use BFS to find the nearest parent method in the inheritance hierarchy
        queue = deque([class_qn])
        visited = {class_qn}  # Don't revisit classes (handle diamond inheritance)

        while queue:
            current_class = queue.popleft()

            # Skip the original class (we're looking for parent methods)
            if current_class != class_qn:
                parent_method_qn = f"{current_class}.{method_name}"

                # Check if this parent class has the method
                if parent_method_qn in self.function_registry:
                    self.ingestor.ensure_relationship_batch(
                        ("Method", "qualified_name", method_qn),
                        "OVERRIDES",
                        ("Method", "qualified_name", parent_method_qn),
                    )
                    logger.debug(
                        f"Method override: {method_qn} OVERRIDES {parent_method_qn}"
                    )
                    return  # Found the nearest override, stop searching

            # Add parent classes to queue for next level of BFS
            if current_class in self.class_inheritance:
                for parent_class_qn in self.class_inheritance[current_class]:
                    if parent_class_qn not in visited:
                        visited.add(parent_class_qn)
                        queue.append(parent_class_qn)

    def _resolve_superclass_from_type_identifier(
            self, type_identifier_node: Node, module_qn: str
    ) -> str | None:
        """Resolve a superclass name from a type_identifier node."""
        parent_text = type_identifier_node.text
        if parent_text:
            parent_name = parent_text.decode("utf8")
            # Resolve to full qualified name if possible
            return (
                    self._resolve_class_name(parent_name, module_qn)
                    or f"{module_qn}.{parent_name}"
            )
        return None

    def _extract_parent_classes(self, class_node: Node, module_qn: str) -> list[str]:
        """Extract parent class names from a class definition."""
        parent_classes = []

        # Look for superclass in Java class definition (extends clause)
        if class_node.type == "class_declaration":
            superclass_node = class_node.child_by_field_name("superclass")
            if superclass_node:
                # Java superclass is a single type identifier
                if superclass_node.type == "type_identifier":
                    resolved_superclass = self._resolve_superclass_from_type_identifier(
                        superclass_node, module_qn
                    )
                    if resolved_superclass:
                        parent_classes.append(resolved_superclass)
                else:
                    # Look for type_identifier children in superclass node
                    for child in superclass_node.children:
                        if child.type == "type_identifier":
                            resolved_superclass = (
                                self._resolve_superclass_from_type_identifier(
                                    child, module_qn
                                )
                            )
                            if resolved_superclass:
                                parent_classes.append(resolved_superclass)
                                break

        return parent_classes

    def _resolve_class_name(self, class_name: str, module_qn: str) -> str | None:
        """Convert a simple class name to its fully qualified name."""
        return resolve_class_name(
            class_name, module_qn, self.import_processor, self.function_registry
        )

    def _is_static_method_in_class(self, method_node: Node) -> bool:
        """Check if this method is a static method inside a class definition."""
        # Check if method has static keyword as sibling
        if method_node.type == "method_definition":
            # Check if any sibling or parent has "static" keyword
            parent = method_node.parent
            if parent and parent.type == "class_body":
                # Look for static keyword in the method definition
                for child in method_node.children:
                    if child.type == "static":
                        return True
        return False

    def _is_method_in_class(self, method_node: Node) -> bool:
        """Check if this method is inside a class definition (static or instance)."""
        # Walk up the tree to see if we're inside a class
        current = method_node.parent
        while current:
            if current.type == "class_body":
                return True
            current = current.parent
        return False

    def _is_inside_method_with_object_literals(self, func_node: Node) -> bool:
        """Check if this function is an object literal method inside a class method."""
        # Walk up to see if we're inside an object literal inside a method_definition
        current = func_node.parent
        found_object = False

        while current:
            if current.type == "object":
                found_object = True
            elif current.type == "method_definition" and found_object:
                # We're inside an object literal inside a method - this should be nested
                return True
            elif current.type == "class_body":
                # Reached class body - stop looking
                break
            current = current.parent

        return False


    def _extract_implemented_interfaces(
            self, class_node: Node, module_qn: str
    ) -> list[str]:
        """Extract implemented interface names from a Java class definition."""
        implemented_interfaces: list[str] = []

        # Look for interfaces field in Java class declaration
        interfaces_node = class_node.child_by_field_name("interfaces")
        if interfaces_node:
            # The interfaces node contains a super_interfaces structure
            # which has a type_list with comma-separated interface types
            self._extract_java_interface_names(
                interfaces_node, implemented_interfaces, module_qn
            )

        return implemented_interfaces

    def _extract_java_interface_names(
            self, interfaces_node: Node, interface_list: list[str], module_qn: str
    ) -> None:
        """Extract interface names from Java interfaces clause using tree-sitter."""
        for child in interfaces_node.children:
            if child.type == "type_list":
                # Type list contains the actual interface types
                for type_child in child.children:
                    if type_child.type == "type_identifier":
                        interface_name = type_child.text
                        if interface_name:
                            interface_name_str = interface_name.decode("utf8")
                            # Resolve to fully qualified name
                            resolved_interface = (
                                    self._resolve_class_name(interface_name_str, module_qn)
                                    or f"{module_qn}.{interface_name_str}"
                            )
                            interface_list.append(resolved_interface)

    def _create_implements_relationship(
            self, class_type: str, class_qn: str, interface_qn: str
    ) -> None:
        """Create an IMPLEMENTS relationship between a class and an interface."""
        self.ingestor.ensure_relationship_batch(
            (class_type, "qualified_name", class_qn),
            "IMPLEMENTS",
            ("Interface", "qualified_name", interface_qn),
        )
