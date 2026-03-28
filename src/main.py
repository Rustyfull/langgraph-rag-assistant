"""
CLI entry point for the LangGraph RAG Assistant.
Run interactively or pass a one-shot question.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.agents import run_query
from src.rag import run_ingestion, vectorstore_exists
from src.utils import  get_logger, get_settings

app = typer.Typer(name="rag-assistant", help="LangGraph RAG Assistant  CLI")
console = Console()
logger = get_logger(__name__)
settings = get_settings()

@app.command()
def query(
        question: str = typer.Argument(None, help="Question to ask. If omitted, enters interactive mode."),
        verbose: bool = typer.Option(False, "--verbose", "-v",help="Show source documents")
):
    """ASk a question to the RAG assistant."""
    _ensure_vectorstore()

    if question:
        _run_single_query(question,verbose)
    else:
        _interactive_mode(verbose)


@app.command()
def ingest(
        force:bool = typer.Option(False,"--force","-f",help="Force re-ingestion")
):
    """Ingest documents into the vector store."""
    console.print("[bold blue]Starting document ingestion...[/]")
    run_ingestion(force=force)
    console.print("[bold green]✓ Ingestion complete[/]")

def _ensure_vectorstore() -> None:
    if not vectorstore_exists():
        console.print("[yellow]No vectore store found - running ingestion first...[/]")
        run_ingestion()



def _run_single_query(question:str, verbose:bool) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        progress.add_task("Thinking...",total=None)
        state = run_query(question)
    _display_result(state,verbose)


def _interactive_mode(verbose:bool) -> None:
    console.print(Panel("[bold]LangGraph RAG Assistant[/] - type [cyan]exit[\] to quit" ,style="blue"))

    while True:
        try:
            question = console.input("[bold cyan]You:[/] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Goodbye![/]")
            break

        if not question or question.lower() in ("exit", "quit", "q"):
            console.print("[yellow]Goodbye![/]")
            break


        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
            console=console
        ) as progress:
            progress.add_task("Processing...",total=None)
            state = run_query(question)

        _display_result(state,verbose)
        console.print()






def _display_result(state:dict, verbose:bool)   -> None:
    answer = state.get("generation","No answer generated")
    console.print("\n[bold green]Assistant:[/]")
    console.print(Markdown(answer))


    if verbose:
        docs = state.get("documents",[])
        if docs:
            console.print(f"\n[dim]Sources ({len(docs)} documents):[/]")
            for i, doc in enumerate(docs,1):
                source = doc.metadata.get("source","unknow")
                console.print(f"    [dim]{i}. {source}[/]")

    retries = state.get("retry_count",0)
    h_score = state.get("hallucination_score","?")
    console.print(
        f"\n[dim]Grounded: {h_score} | Retries: {retries}[/]"
    )

if __name__ == "__main__":
    app()

