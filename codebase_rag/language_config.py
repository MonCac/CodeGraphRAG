from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

COMMON_DECLARATION_IMPORT = ["import_declaration"]


def create_lang_config(**kwargs: Any) -> "LanguageConfig":
    """Helper to create LanguageConfig without redundant name assignment."""
    # Name will be set automatically when configs are processed
    return LanguageConfig(name="", **kwargs)


@dataclass
class LanguageConfig:
    """Configuration for language-specific Tree-sitter parsing."""

    name: str
    file_extensions: list[str]

    # AST node type mappings to semantic concepts
    function_node_types: list[str]
    class_node_types: list[str]
    module_node_types: list[str]
    call_node_types: list[str] = field(default_factory=list)

    # Import statement node types for precise import resolution
    import_node_types: list[str] = field(default_factory=list)
    import_from_node_types: list[str] = field(default_factory=list)

    # Field names for extracting names
    name_field: str = "name"
    body_field: str = "body"

    # Package detection patterns
    package_indicators: list[str] = field(
        default_factory=list
    )  # e.g., ["__init__.py"] for Python

    # Optional pre-formatted Tree-sitter query strings or query generators
    # These override the default node_types-based query generation
    function_query: str | None = None
    class_query: str | None = None
    call_query: str | None = None


######################## Language configurations ###############################
# Automatic generation might add types that are too broad or inaccurate.
# You have to manually check and adjust the configurations after running the
# automatic generation.
################################################################################

LANGUAGE_CONFIGS = {
    "java": create_lang_config(
        file_extensions=[".java"],
        function_node_types=[
            "method_declaration",
            "constructor_declaration",
        ],
        class_node_types=[
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "annotation_type_declaration",
            "record_declaration",
        ],
        module_node_types=["program"],
        package_indicators=[],  # Java uses package declarations
        call_node_types=["method_invocation"],
        import_node_types=COMMON_DECLARATION_IMPORT,
        import_from_node_types=COMMON_DECLARATION_IMPORT,  # Java uses same node for imports
        # Pre-formatted Tree-sitter queries for comprehensive Java parsing
        function_query="""
        (method_declaration
            name: (identifier) @name) @function
        (constructor_declaration
            name: (identifier) @name) @function
        """,
        class_query="""
        (class_declaration
            name: (identifier) @name) @class
        (interface_declaration
            name: (identifier) @name) @class
        (enum_declaration
            name: (identifier) @name) @class
        (annotation_type_declaration
            name: (identifier) @name) @class
        (record_declaration
            name: (identifier) @name) @class
        """,
        call_query="""
        (method_invocation
            name: (identifier) @name) @call
        (object_creation_expression
            type: (type_identifier) @name) @call
        """,
    ),
}


def _initialize_config_names() -> None:
    """Initialize config names based on dict keys."""
    for lang_name, config in LANGUAGE_CONFIGS.items():
        if not config.name:  # Only set if empty (from create_lang_config)
            config.name = lang_name


# Initialize names on module load
_initialize_config_names()


def get_language_config(file_extension: str) -> LanguageConfig | None:
    """Get language configuration based on file extension."""
    for config in LANGUAGE_CONFIGS.values():
        if file_extension in config.file_extensions:
            return config
    return None


def get_language_config_by_name(language_name: str) -> LanguageConfig | None:
    """Get language configuration by language name."""
    return LANGUAGE_CONFIGS.get(language_name.lower())
