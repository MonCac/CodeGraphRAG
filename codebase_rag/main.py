import asyncio
import copy
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any
from datetime import datetime

import typer
from loguru import logger
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import print_formatted_text
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Confirm
from rich.panel import Panel
from rich.text import Text

from codebase_rag.prompts import build_query_question_from_antipattern, build_fix_system_prompt, \
    build_first_fix_user_input, build_fix_user_input
from codebase_rag.services.hierarchical_semantic_builder import HierarchicalSemanticBuilder
from codebase_rag.tools.analyze_antipattern_relevance_files import analyze_files_with_llm
from codebase_rag.tools.graph_extract_query import create_graph_extract_query_tool, query_codebase_with_agent_batch
from .config import (
    EDIT_INDICATORS,
    EDIT_REQUEST_KEYWORDS,
    EDIT_TOOLS,
    ORANGE_STYLE,
    detect_provider_from_model,
    settings, resolve_output_path,
)
from .graph_updater import GraphProjectUpdater, MemgraphIngestor, GraphAntipatternUpdater
from .parser_loader import load_parsers
from .services.llm import CypherGenerator, create_rag_orchestrator, create_repair_code_model
from .tools.code_retrieval import CodeRetriever, create_code_retrieval_tool
from .tools.codebase_query import create_query_tool
from .tools.directory_lister import DirectoryLister, create_directory_lister_tool
from .tools.document_analyzer import DocumentAnalyzer, create_document_analyzer_tool
from .tools.file_editor import FileEditor, create_file_editor_tool
from .tools.file_reader import FileReader, create_file_reader_tool
from .tools.file_writer import FileWriter, create_file_writer_tool
from .tools.shell_command import ShellCommander, create_shell_command_tool

# Style constants
confirm_edits_globally = True

# Pre-compile regex patterns
_FILE_MODIFICATION_PATTERNS = [
    re.compile(
        r"(modified|updated|created|edited):\s*[\w/\\.-]+\.(py|js|ts|java|cpp|c|h|go|rs)"
    ),
    re.compile(
        r"file\s+[\w/\\.-]+\.(py|js|ts|java|cpp|c|h|go|rs)\s+(modified|updated|created|edited)"
    ),
    re.compile(r"writing\s+to\s+[\w/\\.-]+\.(py|js|ts|java|cpp|c|h|go|rs)"),
]

app = typer.Typer(
    name="graph-code",
    help="An accurate Retrieval-Augmented Generation (RAG) system that analyzes "
         "multi-language codebases using Tree-sitter, builds comprehensive knowledge "
         "graphs, and enables natural language querying of codebase structure and "
         "relationships.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console(width=None, force_terminal=True)
# Session logging
session_log_file = None
session_cancelled = False

# Global flag to control edit confirmation
confirm_edits = True


def init_session_log(project_root: Path) -> Path:
    """Initialize session log file."""
    global session_log_file
    log_dir = project_root / "tmp"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_log_file = log_dir / f"session_{timestamp}_{uuid.uuid4().hex[:8]}.log"
    with open(session_log_file, "w") as f:
        f.write("=== CODE-GRAPH RAG SESSION LOG ===\n\n")
    return session_log_file


def log_session_event(event: str) -> None:
    """Log an event to the session file."""
    global session_log_file
    if session_log_file:
        with open(session_log_file, "a") as f:
            f.write(f"{event}\n")


def get_session_context() -> str:
    """Get the full session context for cancelled operations."""
    global session_log_file
    if session_log_file and session_log_file.exists():
        content = Path(session_log_file).read_text()
        return f"\n\n[SESSION CONTEXT - Previous conversation in this session]:\n{content}\n[END SESSION CONTEXT]\n\n"
    return ""


def is_edit_operation_request(question: str) -> bool:
    """Check if the user's question/request would likely result in edit operations."""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in EDIT_REQUEST_KEYWORDS)


async def _handle_rejection(
        rag_agent: Any, message_history: list[Any], console: Console
) -> Any:
    """Handle user rejection of edits with agent acknowledgment."""
    rejection_message = "The user has rejected the changes that were made. Please acknowledge this and consider if any changes need to be reverted."

    with console.status("[bold yellow]Processing rejection...[/bold yellow]"):
        rejection_response = await run_with_cancellation(
            console,
            rag_agent.run(rejection_message, message_history=message_history),
        )

    if not (
            isinstance(rejection_response, dict) and rejection_response.get("cancelled")
    ):
        rejection_markdown = Markdown(rejection_response.output)
        logger.info(
            Panel(
                rejection_markdown,
                title="[bold yellow]Response to Rejection[/bold yellow]",
                border_style="yellow",
            )
        )
        message_history.extend(rejection_response.new_messages())

    return rejection_response


def is_edit_operation_response(response_text: str) -> bool:
    """Enhanced check if the response contains edit operations that need confirmation."""
    response_lower = response_text.lower()

    # Check for tool usage
    tool_usage = any(tool in response_lower for tool in EDIT_TOOLS)

    # Check for content indicators
    content_indicators = any(
        indicator in response_lower for indicator in EDIT_INDICATORS
    )

    # Check for regex patterns
    pattern_match = any(
        pattern.search(response_lower) for pattern in _FILE_MODIFICATION_PATTERNS
    )

    return tool_usage or content_indicators or pattern_match


def _handle_chat_images(question: str, project_root: Path) -> str:
    """
    Checks for image file paths in the question, copies them to a temporary
    directory, and replaces the path in the question.
    """
    # Use shlex to properly parse the question and handle escaped spaces
    try:
        tokens = shlex.split(question)
    except ValueError:
        # Fallback to simple split if shlex fails
        tokens = question.split()

    # Find image files in tokens
    image_extensions = (".png", ".jpg", ".jpeg", ".gif")
    image_files = [
        token
        for token in tokens
        if token.startswith("/") and token.lower().endswith(image_extensions)
    ]

    if not image_files:
        return question

    updated_question = question
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    for original_path_str in image_files:
        original_path = Path(original_path_str)

        if not original_path.exists() or not original_path.is_file():
            logger.warning(f"Image path found, but does not exist: {original_path_str}")
            continue

        try:
            new_path = tmp_dir / f"{uuid.uuid4()}-{original_path.name}"
            shutil.copy(original_path, new_path)
            new_relative_path = new_path.relative_to(project_root)

            # Find and replace all possible quoted/escaped versions of this path
            # Try different forms the path might appear in the original question
            path_variants = [
                # Backslash-escaped spaces: /path/with\ spaces.png
                original_path_str.replace(" ", r"\ "),
                # Single quoted: '/path/with spaces.png'
                f"'{original_path_str}'",
                # Double quoted: "/path/with spaces.png"
                f'"{original_path_str}"',
                # Unquoted: /path/with spaces.png
                original_path_str,
            ]

            # Try each variant and replace if found
            replaced = False
            for variant in path_variants:
                if variant in updated_question:
                    updated_question = updated_question.replace(
                        variant, str(new_relative_path)
                    )
                    replaced = True
                    break

            if not replaced:
                logger.warning(
                    f"Could not find original path in question for replacement: {original_path_str}"
                )

            logger.info(f"Copied image to temporary path: {new_relative_path}")
        except Exception as e:
            logger.error(f"Failed to copy image to temporary directory: {e}")

    return updated_question


async def run_with_cancellation(
        console: Console, coro: Any, timeout: float | None = None
) -> Any:
    """Run a coroutine with proper Ctrl+C cancellation that doesn't exit the program."""
    task = asyncio.create_task(coro)

    try:
        return await asyncio.wait_for(task, timeout=timeout) if timeout else await task
    except TimeoutError:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        logger.info(
            f"\n[bold yellow]Operation timed out after {timeout} seconds.[/bold yellow]"
        )
        return {"cancelled": True, "timeout": True}
    except (asyncio.CancelledError, KeyboardInterrupt):
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        logger.info("\n[bold yellow]Thinking cancelled.[/bold yellow]")
        return {"cancelled": True}


def _setup_common_initialization(repo_path: str) -> Path:
    """Common setup logic for both main and optimize functions."""
    # Logger initialization
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {message}",
        colorize=True  # ✅ 关键，启用彩色解析
    )
    # Temporary directory setup (keep existing tmp if already present)
    project_root = Path(repo_path).resolve()
    tmp_dir = project_root / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    return project_root


def _create_configuration_table(
        repo_path: str,
        title: str = "Graph-Code Initializing...",
        language: str | None = None,
) -> Table:
    """Create and return a configuration table."""
    table = Table(title=f"[bold green]{title}[/bold green]")
    table.add_column("Configuration", style="cyan")
    table.add_column("Value", style="magenta")

    # Add language row if provided (for optimization sessions)
    if language:
        table.add_row("Target Language", language)

    orchestrator_model = settings.active_orchestrator_model
    orchestrator_provider = detect_provider_from_model(orchestrator_model)
    table.add_row(
        "Orchestrator Model", f"{orchestrator_model} ({orchestrator_provider})"
    )

    cypher_model = settings.active_cypher_model
    cypher_provider = detect_provider_from_model(cypher_model)
    table.add_row("Cypher Model", f"{cypher_model} ({cypher_provider})")

    embedding_model = settings.active_embedding_model
    embedding_provider = detect_provider_from_model(embedding_model)
    table.add_row("Cypher Model", f"{embedding_model} ({embedding_provider})")

    # Show local endpoint if any model is using local provider
    if orchestrator_provider == "local" or cypher_provider == "local":
        table.add_row("Local Model Endpoint", str(settings.LOCAL_MODEL_ENDPOINT))

    # Show edit confirmation status
    confirmation_status = (
        "Enabled" if confirm_edits_globally else "Disabled (YOLO Mode)"
    )
    table.add_row("Edit Confirmation", confirmation_status)
    table.add_row("Target Repository", repo_path)

    return table


def _update_model_settings(
        orchestrator_model: str | None,
        cypher_model: str | None,
        embedding_model: str | None,
) -> None:
    """Update model settings based on command-line arguments."""
    # Set orchestrator model if provided
    if orchestrator_model:
        settings.set_orchestrator_model(orchestrator_model)

    # Set cypher model if provided
    if cypher_model:
        settings.set_cypher_model(cypher_model)

    # Set embedding model if provided
    if embedding_model:
        settings.set_embedding_model(embedding_model)


def get_multiline_input(prompt_text: str = "Ask a question") -> str:
    """Get multiline input from user with Ctrl+J to submit."""
    bindings = KeyBindings()

    @bindings.add("c-j")
    def submit(event: Any) -> None:
        """Submit the current input."""
        event.app.exit(result=event.app.current_buffer.text)

    @bindings.add("enter")
    def new_line(event: Any) -> None:
        """Insert a new line instead of submitting."""
        event.current_buffer.insert_text("\n")

    @bindings.add("c-c")
    def keyboard_interrupt(event: Any) -> None:
        """Handle Ctrl+C."""
        event.app.exit(exception=KeyboardInterrupt)

    # Convert Rich markup to plain text using Rich's parser
    clean_prompt = Text.from_markup(prompt_text).plain

    # Display the colored prompt first
    print_formatted_text(
        HTML(
            f"<ansigreen><b>{clean_prompt}</b></ansigreen> <ansiyellow>(Press Ctrl+J to submit, Enter for new line)</ansiyellow>: "
        )
    )

    # Use simple prompt without formatting to avoid alignment issues
    result = prompt(
        "",
        multiline=True,
        key_bindings=bindings,
        wrap_lines=True,
        style=ORANGE_STYLE,
    )
    if result is None:
        raise EOFError
    return result.strip()  # type: ignore[no-any-return]


async def run_chat_loop(
        rag_agent: Any, message_history: list[Any], project_root: Path
) -> None:
    """Runs the main chat loop with proper edit confirmation."""
    global session_cancelled

    # Initialize session logging
    init_session_log(project_root)

    while True:
        try:
            # Get user input
            '''
            # true input
            question = await asyncio.to_thread(
                get_multiline_input, "[bold cyan]Ask a question[/bold cyan]"
            )
            '''

            # DEBUG: 固定用户输入
            question = "介绍这个repo的结构、功能、内容"
            # question = "这是一个测试问题"
            if question.lower() in ["exit", "quit"]:
                break
            if not question.strip():
                continue

            # Log user question
            log_session_event(f"USER: {question}")

            # If previous thinking was cancelled, add session context
            if session_cancelled:
                question_with_context = question + get_session_context()
                session_cancelled = False
            else:
                question_with_context = question

            # Handle images in the question
            question_with_context = _handle_chat_images(
                question_with_context, project_root
            )

            # Check if this might be an edit operation and warn user upfront
            might_edit = is_edit_operation_request(question)
            if confirm_edits_globally and might_edit:
                logger.info(
                    "\n[bold yellow]⚠️  This request might result in file modifications.[/bold yellow]"
                )
                if not Confirm.ask(
                        "[bold cyan]Do you want to proceed with this request?[/bold cyan]"
                ):
                    logger.info("[bold red]❌ Request cancelled by user.[/bold red]")
                    continue

            with console.status(
                    "[bold green]Thinking... (Press Ctrl+C to cancel)[/bold green]"
            ):
                response = await run_with_cancellation(
                    console,
                    rag_agent.run(
                        question_with_context, message_history=message_history
                    ),
                )

                if isinstance(response, dict) and response.get("cancelled"):
                    log_session_event("ASSISTANT: [Thinking was cancelled]")
                    session_cancelled = True
                    continue

            # Display the response
            markdown_response = Markdown(response.output)
            logger.info(
                Panel(
                    markdown_response,
                    title="[bold green]Assistant[/bold green]",
                    border_style="green",
                )
            )

            # Check if the response actually contains edit operations
            if confirm_edits_globally and is_edit_operation_response(response.output):
                logger.info(
                    "\n[bold yellow]⚠️  The assistant has performed file modifications.[/bold yellow]"
                )

                if not Confirm.ask(
                        "[bold cyan]Do you want to keep these changes?[/bold cyan]"
                ):
                    logger.info("[bold red]❌ User rejected the changes.[/bold red]")
                    await _handle_rejection(rag_agent, message_history, console)
                    continue
                else:
                    logger.info(
                        "[bold green]✅ Changes accepted by user.[/bold green]"
                    )

            # Log assistant response
            log_session_event(f"ASSISTANT: {response.output}")

            # Add the response to message history only if it wasn't rejected
            message_history.extend(response.new_messages())

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error("An unexpected error occurred: {}", e, exc_info=True)
            logger.info(f"[bold red]An unexpected error occurred: {e}[/bold red]")
            traceback.print_exc()


def _export_graph_to_file(ingestor: MemgraphIngestor, output: str) -> bool:
    """
    Export graph data to a JSON file.

    Args:
        ingestor: The MemgraphIngestor instance to export from
        output: Output file path

    Returns:
        True if export was successful, False otherwise
    """

    try:
        graph_data = ingestor.export_graph_to_dict()
        output_path = Path(output)

        # Ensure the output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write JSON with proper formatting
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"[bold green]Graph exported successfully to: {output_path.absolute()}[/bold green]"
        )
        logger.info(
            f"[bold cyan]Export contains {graph_data['metadata']['total_nodes']} nodes and {graph_data['metadata']['total_relationships']} relationships[/bold cyan]"
        )
        return True

    except Exception as e:
        logger.info(f"[bold red]Failed to export graph: {e}[/bold red]")
        logger.error(f"Export error: {e}", exc_info=True)
        return False


def save_semantic_results(result: dict, save_dir: str = "tmp") -> None:
    """
    将 build_node_semantics 的结果分别保存为 total_result.json 和 file_result.json。

    Args:
        result (dict): 包含 'texts', 'ids', 'metadatas'
        save_dir (str): 保存的目录路径，例如 "D:/data/"
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    semantic_total_path = save_dir / "semantic_total_result.json"
    semantic_file_path = save_dir / "semantic_file_result.json"

    try:
        # ✅ 保存 total_result.json
        with open(semantic_total_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        logger.success(f"✅ 总体语义结果已保存到: {semantic_total_path}")

        metadatas = result.get("metadatas", [])
        # 先筛选所有 labels 不含 File 的节点，方便匹配 parentId
        other_nodes = [m for m in metadatas if "File" not in m.get("labels", [])]

        filtered_metadatas = []
        for file_node in [m for m in metadatas if "File" in m.get("labels", [])]:
            file_id = file_node.get("properties", {}).get("id")
            if file_id is None:
                continue

            # 找 parentId == file_id 且 labels 中含 File 的第一个节点
            matched_file = None
            for node in other_nodes:
                if node.get("properties", {}).get("parentId") == file_id and "File" in node.get("properties", []):
                    matched_file = node
                    break

            if matched_file:
                new_node = copy.deepcopy(file_node)
                # 只复制 matched_file["properties"]["File"] 的内容给 new_node["properties"]["File"]
                file_value = matched_file.get("properties", {}).get("File")
                if file_value is not None:
                    new_node["properties"]["File"] = file_value
                filtered_metadatas.append(new_node)
            # 如果没找到匹配，跳过该节点，不添加

        file_result = {
            "metadatas": filtered_metadatas
        }

        # ✅ 保存 file_result.json
        with open(semantic_file_path, "w", encoding="utf-8") as f:
            json.dump(file_result, f, ensure_ascii=False, indent=4)

        logger.success(f"✅ 文件级语义结果已保存到: {semantic_file_path}（共 {len(filtered_metadatas)} 个文件节点）")

    except Exception as e:
        logger.error(f"❌ 保存语义结果失败: {e}")


def save_antipattern_relevance_result(result, file_path):
    """
    保存反模式相关性结果到 JSON 文件

    参数:
        result: 可序列化为 JSON 的数据
        file_path: 保存的目标路径
    """
    # 确保目录存在
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # 写入 JSON 文件
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    logger.info(f"[bold green]保存成功:[/bold green] {file_path}")

    return file_path


def _initialize_services_and_agent(repo_path: str, ingestor: MemgraphIngestor) -> Any:
    """Initializes all services and creates the RAG agent."""
    # Validate settings once before initializing any LLM services
    settings.validate_for_usage()

    cypher_generator = CypherGenerator()
    code_retriever = CodeRetriever(project_root=repo_path, ingestor=ingestor)
    file_reader = FileReader(project_root=repo_path)
    file_writer = FileWriter(project_root=repo_path)
    file_editor = FileEditor(project_root=repo_path)
    shell_commander = ShellCommander(
        project_root=repo_path, timeout=settings.SHELL_COMMAND_TIMEOUT
    )
    directory_lister = DirectoryLister(project_root=repo_path)
    document_analyzer = DocumentAnalyzer(project_root=repo_path)

    query_tool = create_query_tool(ingestor, cypher_generator, console)
    code_tool = create_code_retrieval_tool(code_retriever)
    file_reader_tool = create_file_reader_tool(file_reader)
    file_writer_tool = create_file_writer_tool(file_writer)
    file_editor_tool = create_file_editor_tool(file_editor)
    shell_command_tool = create_shell_command_tool(shell_commander)
    directory_lister_tool = create_directory_lister_tool(directory_lister)
    document_analyzer_tool = create_document_analyzer_tool(document_analyzer)

    rag_agent = create_rag_orchestrator(
        tools=[
            query_tool,
            code_tool,
            file_reader_tool,
            file_writer_tool,
            file_editor_tool,
            shell_command_tool,
            directory_lister_tool,
            document_analyzer_tool,
        ]
    )
    return rag_agent


def _initialize_graph_extract_service_and_agent(ingestor: MemgraphIngestor) -> Any:
    """Initializes graph extract service and creates the RAG agent."""
    # Validate settings once before initializing any LLM services
    settings.validate_for_usage()

    cypher_generator = CypherGenerator()

    query_codebase_knowledge_graph_tool = create_graph_extract_query_tool(ingestor, cypher_generator, console)

    rag_agent = create_rag_orchestrator(
        tools=[
            query_codebase_knowledge_graph_tool,
        ]
    )
    return rag_agent


async def main_async(repo_path: str) -> None:
    """Initializes services and runs the main application loop."""
    project_root = _setup_common_initialization(repo_path)

    table = _create_configuration_table(repo_path)
    logger.info(table)

    with MemgraphIngestor(
            host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
    ) as ingestor:
        logger.info("[bold green]Successfully connected to Memgraph.[/bold green]")
        logger.info(
            Panel(
                "[bold yellow]Ask questions about your codebase graph. Type 'exit' or 'quit' to end.[/bold yellow]",
                border_style="yellow",
            )
        )

        rag_agent = _initialize_services_and_agent(repo_path, ingestor)
        await run_chat_loop(rag_agent, [], project_root)


@app.command()
def start(
        repo_path: str | None = typer.Option(
            None, "--repo-path", help="Path to the target repository for code retrieval"
        ),
        antipattern_relation_path: str | None = typer.Option(
            None, "--antipattern-relation-path", help="The path to the code information related to the anti-pattern"
        ),
        update_project_graph: bool = typer.Option(
            False,
            "--update-project-graph",
            help="Update the knowledge graph by parsing the repository",
        ),
        update_antipattern_graph: bool = typer.Option(
            True,
            "--update-antipattern-graph",
            help="Update the knowledge graph by parsing the repository and antipattern files",
        ),
        antipattern_type: str = typer.Option(
            "ch",
            "--antipattern-type",
            help="Enable hybrid retrieval (graph + embedding) during query.",
        ),
        semantic_enhance: bool = typer.Option(
            True,
            "--semantic-enhance",
            help="Generate semantic analyses and embeddings for entities after graph update.",
        ),
        hybrid_query: bool = typer.Option(
            False,
            "--hybrid-query",
            help="Enable hybrid retrieval (graph + embedding) during query.",
        ),
        clean: bool = typer.Option(
            True,
            "--clean",
            help="Clean the database before updating (use when adding first repo)",
        ),
        output: str | None = typer.Option(
            "tmp/final-result.json",
            "-o",
            "--output",
            help="Export graph to JSON file after updating (requires --update-graph)",
        ),
        orchestrator_model: str | None = typer.Option(
            None, "--orchestrator-model", help="Specify the orchestrator model ID"
        ),
        cypher_model: str | None = typer.Option(
            None, "--cypher-model", help="Specify the Cypher generator model ID"
        ),
        embedding_model: str | None = typer.Option(
            None, "--embedding-model", help="Specify the Embedding generator model ID"
        ),
        no_confirm: bool = typer.Option(
            False,
            "--no-confirm",
            help="Disable confirmation prompts for edit operations (YOLO mode)",
        ),
) -> None:
    """Starts the Codebase RAG CLI."""
    global confirm_edits_globally
    # restart_memgraph()

    # Set confirmation mode based on flag
    confirm_edits_globally = not no_confirm

    target_repo_path = repo_path or settings.TARGET_REPO_PATH
    antipattern_relation_path = antipattern_relation_path or settings.ANTIPATTERN_RELATION_PATH

    # Validate output option usage
    # if output and not update_graph:
    #     logger.info(
    #         "[bold red]Error: --output/-o option requires --update-graph to be specified.[/bold red]"
    #     )
    #     raise typer.Exit(1)

    _update_model_settings(orchestrator_model, cypher_model, embedding_model)

    if update_project_graph:
        repo_to_update = Path(target_repo_path)
        logger.info(
            f"[bold green]Updating knowledge graph for: {repo_to_update}[/bold green]"
        )

        with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
        ) as ingestor:
            if clean:
                logger.info("[bold yellow]Cleaning database...[/bold yellow]")
                ingestor.clean_database()
            ingestor.ensure_constraints()

            # # Load parsers and queries
            # parsers, queries = load_parsers()
            #
            # updater = GraphProjectUpdater(ingestor, repo_to_update)
            # updater.run()

            if semantic_enhance:
                logger.info("[bold blue]Generating semantic embeddings...[/bold blue]")
                # 本地测试用
                json_path = Path("tmp\\algorithm-graph.json")  # 替换为你的文件路径
                if not json_path.exists():
                    raise FileNotFoundError(f"文件不存在: {json_path}")

                with open(json_path, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)

                # 实际执行逻辑
                # graph_data = ingestor.export_graph_to_dict()

                # output_file = Path(output)
                # output_dir = output_file.parent
                # hierarchical_semantic_builder = HierarchicalSemanticBuilder()
                # result = hierarchical_semantic_builder.build_node_semantics(graph_data)
                # save_semantic_results(result, "CodeGraphRAG\\tmp")

                logger.info("[bold green]Semantic embeddings generation completed![/bold green]")

        logger.info("[bold green]Graph update completed![/bold green]")

    if update_antipattern_graph:
        repo_to_update = Path(target_repo_path)
        antipattern_to_update = Path(antipattern_relation_path)
        logger.info(
            f"[bold green]Updating knowledge graph for: {repo_to_update} \n and \n {antipattern_to_update}[/bold green]"
        )
        with MemgraphIngestor(
                host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
        ) as ingestor:
            if clean:
                logger.info("[bold yellow]Cleaning database...[/bold yellow]")
                ingestor.clean_database()
            ingestor.ensure_constraints()

            # # Load parsers and queries
            parsers, queries = load_parsers()

            updater = GraphAntipatternUpdater(ingestor, repo_to_update, antipattern_to_update)
            updater.run()
            _export_graph_to_file(ingestor, output)

            # 与 LLM 交互，完成第一层文件的提取
            # 仿照 run_chat_loop 构建 chat，完成对数据库的内容提取。输入是反模式的具体体现的文件，用它来构建 prompt。输出就是对知识图谱的提取结果，存储为 json
            # 然后提供给 semantic_enhance。让 semantic_enhance 生成之后再存入数据库。
            # asyncio.run(run_multi_interaction(ingestor, antipattern_relation_path))
            # 可能遇到的问题：
            # 1. 数据库最新存储时需要让每次的 node_id 的 start 为 0
            # 2. 对于 semantic_enhance 的内容，如何存储，可以进行辨识。不删除原来的内容。但又想让 from_id 和 to_id 不冲突，对应的仍然是 id。

            if semantic_enhance:
                logger.info("[bold blue]Generating semantic embeddings...[/bold blue]")
                # # 本地测试用
                # json_path = Path("tmp\\algorithm-graph.json")  # 替换为你的文件路径
                # if not json_path.exists():
                #     raise FileNotFoundError(f"文件不存在: {json_path}")
                # with open(json_path, "r", encoding="utf-8") as f:
                #     graph_data = json.load(f)

                # 实际执行逻辑
                graph_data = ingestor.export_graph_to_dict()

                output_file = Path(output)
                output_dir = output_file.parent
                hierarchical_semantic_builder = HierarchicalSemanticBuilder()
                result = hierarchical_semantic_builder.build_node_semantics(graph_data)
                save_semantic_results(result, "tmp")

                logger.info("[bold green]Semantic embeddings generation completed![/bold green]")

                # 编写一个与 llm 交互的函数。
                # 输入是tmp/file_result.json，都是file级别的。和.env中的antipattern路径，构建反模式代码和反模式描述的json拼接prompt，来判断file_result.json中的每个文件是否与反模式修复相关。
                # 得到最终的files
                result = analyze_files_with_llm(
                    os.path.join("tmp", "semantic_file_result.json"),
                    antipattern_relation_path
                )

                related_files_json_path = save_antipattern_relevance_result(
                    result,
                    os.path.join("tmp", "antipattern_relevance_result.json")
                )
                # 构建最终的修复架构反模式的与 llm 交互的流程
                # 1. System Prompt 中包含的内容：背景、反模式描述、修复方法、修复案例
                # 2. Input 包含的内容：当前反模式的 json 文件，反模式具体代码，相关文件与反模式文件调用关系相关代码。
                # 3. Output 包含的内容：对反模式文件的修复描述，具体代码修复
                # 4. 多次交互，完善其他文件的代码具体内容
                # 第三个参数是上面生成的路径json，所以需要统一路径
                result = asyncio.run(run_repair_code_llm(antipattern_type, antipattern_relation_path,
                                                         "tmp/antipattern_relevance_result.json"))

                save_repair_results_to_json(result, "./repair_outputs")

    # Export graph if output file specified
    if output:
        output = resolve_output_path(output)
        logger.info(f"[bold cyan]Exporting graph to: {output}[/bold cyan]")
        if not _export_graph_to_file(ingestor, output):
            raise typer.Exit(1)

    logger.info("[bold green]Graph update completed![/bold green]")


# try:
#     asyncio.run(main_async(target_repo_path))
# except KeyboardInterrupt:
#     logger.info("\n[bold red]Application terminated by user.[/bold red]")
# except ValueError as e:
#     logger.info(f"[bold red]Startup Error: {e}[/bold red]")


async def run_multi_interaction(ingestor: MemgraphIngestor, antipattern_relation_path: str):
    """Runs multi-turn LLM interaction where agent can autonomously call tools."""

    # 初始化 agent，只注册工具
    rag_agent = _initialize_graph_extract_service_and_agent(ingestor)

    # 构造初始问题
    try:
        user_question = build_query_question_from_antipattern(antipattern_relation_path)
    except FileNotFoundError:
        user_question = "获取全部实体"
    logger.info(f"[bold cyan]User question:[/bold cyan] {user_question}")

    # 控制台
    console = Console(width=None, force_terminal=True)

    # 第一次交互：LLM生成自然语言查询
    try:
        console.print("[bold green]LLM generating query instruction...[/bold green]")
        llm_response = await rag_agent.run(user_question)
        # 兼容不同返回类型
        query_instruction = getattr(llm_response, "output", str(llm_response)).strip()
        console.print(
            Panel(
                Markdown(query_instruction),
                title="[bold green]LLM Generated Instruction[/bold green]",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[red]Error during LLM instruction generation:[/red] {e}")
        traceback.print_exc()
        return str(e)

    # 第二次交互：调用 query_codebase_with_agent_batch 批量执行查询
    try:
        console.print("[bold green]Executing batch query with agent...[/bold green]")
        results = await query_codebase_with_agent_batch(
            user_input=query_instruction,
            ingestor=ingestor,
            console=console
        )
        console.print(f"[bold green]Batch query completed. {len(results)} results.[/bold green]")
    except Exception as e:
        console.print(f"[red]Error during batch query execution:[/red] {e}")
        traceback.print_exc()
        results = []

    return results


async def async_run_with_retry(func, *args, max_retries=2, retry_interval=1, **kwargs):
    """
    异步调用重试封装，最多重试 max_retries 次。
    retry_interval：每次重试间隔秒数。
    func: 需要重试的异步函数
    args, kwargs: 传给 func 的参数
    """
    attempt = 0
    while attempt <= max_retries:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise e
            else:
                logger.warning(f"异步调用失败，第 {attempt} 次重试，错误：{e}")
                await asyncio.sleep(retry_interval)


async def run_repair_code_llm(antipattern_type: str, antipattern_to_update: str, related_files_json_path: str):
    """Runs multi-turn LLM interaction where agent can autonomously call tools."""

    rag_agent = create_repair_code_model(build_fix_system_prompt(antipattern_type))
    console = Console(width=None, force_terminal=True)

    try:
        user_question = build_first_fix_user_input(antipattern_to_update, related_files_json_path)
    except FileNotFoundError:
        user_question = ""
    logger.info(f"[bold cyan]User question:[/bold cyan] {user_question}")

    # 第一次交互
    try:
        console.print("[bold green]LLM generating first fix answer...[/bold green]")
        llm_response = await async_run_with_retry(rag_agent.run, user_question, max_retries=2)
        first_llm_output_raw = getattr(llm_response, "output", str(llm_response)).strip()

        console.print(
            Panel(
                Markdown(first_llm_output_raw),
                title="[bold green]LLM Generated Instruction[/bold green]",
                border_style="green",
            )
        )

        # 尝试转成 dict，如果是字符串或者其它格式，按你需求调整
        first_llm_output = first_llm_output_raw
        if isinstance(first_llm_output_raw, str):
            import json
            try:
                first_llm_output = json.loads(first_llm_output_raw)
            except Exception:
                # 无法解析为 JSON，保持原样
                pass

    except Exception as e:
        console.print(f"[red]Error during LLM instruction generation:[/red] {e}")
        traceback.print_exc()
        return str(e)

    overall_desc = first_llm_output.get("overall_repair_description", "")
    file_descriptions = first_llm_output.get("file_repair_descriptions", [])

    results = []

    # 第二次多轮交互
    for file_entry in file_descriptions:
        user_input = build_fix_user_input(overall_desc, file_entry)
        try:
            console.print("[bold green]LLM generating fix answer for a file...[/bold green]")
            second_llm_response = await async_run_with_retry(rag_agent.run, user_input, max_retries=2)
            second_llm_output = getattr(second_llm_response, "output", str(second_llm_response)).strip()

            console.print(
                Panel(
                    Markdown(second_llm_output),
                    title="[bold green]LLM Generated Instruction[/bold green]",
                    border_style="green",
                )
            )

            result_entry = {
                "file": file_entry.get("file", ""),
                "source": file_entry.get("source", ""),
                "repair_description": file_entry.get("repair_description", ""),
                "repair_code": second_llm_output,
            }
            results.append(result_entry)

        except Exception as e:
            console.print(f"[red]Error during LLM instruction generation:[/red] {e}")
            traceback.print_exc()
            # 这里不返回错误，继续执行下一条，或者你可以改成记录失败条目
            # return str(e)  # 如果你要失败立即返回，取消注释这一行

    return results


def save_repair_results_to_json(results: list, output_dir: str, filename: str = None):
    """
    将 run_repair_code_llm 的返回结果保存为 JSON 文件。

    Args:
        results (list): run_repair_code_llm 的返回结果（list[dict]）
        output_dir (str): 保存目录路径
        filename (str, optional): 输出文件名（不含路径）。默认为 "repair_results_{timestamp}.json"

    Returns:
        str: 保存的 JSON 文件完整路径
    """
    if not results:
        raise ValueError("results 为空，无法保存。")

    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 默认文件名
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"repair_results_{timestamp}.json"

    output_path = os.path.join(output_dir, filename)

    # 保存 JSON 文件
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"[✅] 修复结果已成功保存至：{output_path}")
    except Exception as e:
        print(f"[❌] 保存 JSON 文件失败：{e}")
        raise

    return output_path


def restart_memgraph():
    try:
        # 尝试删除容器，忽略报错
        subprocess.run(['docker', 'rm', '-f', 'memgraph'], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # 启动新的容器
        subprocess.run([
            'docker', 'run', '-d',
            '-p', '7687:7687',
            '-p', '7444:7444',
            '--name', 'memgraph',
            'memgraph/memgraph-mage:latest'
        ], check=True)
        print("Memgraph container restarted successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    app()
