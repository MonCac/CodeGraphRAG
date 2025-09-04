import asyncio
import json
import re
import shlex
import shutil
import sys
import uuid
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from .config import (
    EDIT_INDICATORS,
    EDIT_REQUEST_KEYWORDS,
    EDIT_TOOLS,
    ORANGE_STYLE,
    detect_provider_from_model,
    settings,
)
# from .graph_updater import GraphUpdater, MemgraphIngestor


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


# # Global flag to control edit confirmation
# confirm_edits = True


def _update_model_settings(
    orchestrator_model: str | None,
    cypher_model: str | None,
) -> None:
    """Update model settings based on command-line arguments."""
    # Set orchestrator model if provided
    if orchestrator_model:
        settings.set_orchestrator_model(orchestrator_model)

    # Set cypher model if provided
    if cypher_model:
        settings.set_cypher_model(cypher_model)


async def main_async(repo_path: str) -> None:
    """Initializes services and runs the main application loop."""
    # project_root = _setup_common_initialization(repo_path)
    #
    # table = _create_configuration_table(repo_path)
    # console.print(table)
    #
    # with MemgraphIngestor(
    #     host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
    # ) as ingestor:
    #     console.print("[bold green]Successfully connected to Memgraph.[/bold green]")
    #     console.print(
    #         Panel(
    #             "[bold yellow]Ask questions about your codebase graph. Type 'exit' or 'quit' to end.[/bold yellow]",
    #             border_style="yellow",
    #         )
    #     )
    #
    #     rag_agent = _initialize_services_and_agent(repo_path, ingestor)
    #     await run_chat_loop(rag_agent, [], project_root)


@app.command()
def start(
        repo_path: str | None = typer.Option(
            None, "--repo-path", help="Path to the target repository for code retrieval"
        ),
        update_graph: bool = typer.Option(
            False,
            "--update-graph",
            help="Update the knowledge graph by parsing the repository",
        ),
        clean: bool = typer.Option(
            False,
            "--clean",
            help="Clean the database before updating (use when adding first repo)",
        ),
        output: str | None = typer.Option(
            None,
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
        no_confirm: bool = typer.Option(
            False,
            "--no-confirm",
            help="Disable confirmation prompts for edit operations (YOLO mode)",
        ),
) -> None:
    """Starts the Codebase RAG CLI."""
    global confirm_edits_globally

    # Set confirmation mode based on flag
    confirm_edits_globally = not no_confirm

    target_repo_path = repo_path or settings.TARGET_REPO_PATH

    # Validate output option usage
    if output and not update_graph:
        console.print(
            "[bold red]Error: --output/-o option requires --update-graph to be specified.[/bold red]"
        )
        raise typer.Exit(1)

    _update_model_settings(orchestrator_model, cypher_model)

    if update_graph:
        repo_to_update = Path(target_repo_path)
        console.print(
            f"[bold green]Updating knowledge graph for: {repo_to_update}[/bold green]"
        )

        # with MemgraphIngestor(
        #         host=settings.MEMGRAPH_HOST, port=settings.MEMGRAPH_PORT
        # ) as ingestor:
        #     if clean:
        #         console.print("[bold yellow]Cleaning database...[/bold yellow]")
        #         ingestor.clean_database()
        #     ingestor.ensure_constraints()
        #
        #     # Load parsers and queries
        #     parsers, queries = load_parsers()
        #
        #     updater = GraphUpdater(ingestor, repo_to_update, parsers, queries)
        #     updater.run()
        #
        #     # Export graph if output file specified
        #     if output:
        #         console.print(f"[bold cyan]Exporting graph to: {output}[/bold cyan]")
        #         if not _export_graph_to_file(ingestor, output):
        #             raise typer.Exit(1)

        console.print("[bold green]Graph update completed![/bold green]")
        return

    try:
        asyncio.run(main_async(target_repo_path))
    except KeyboardInterrupt:
        console.print("\n[bold red]Application terminated by user.[/bold red]")
    except ValueError as e:
        console.print(f"[bold red]Startup Error: {e}[/bold red]")


if __name__ == "__main__":
    app()
