# test_typer.py
import typer
import sys

app = typer.Typer()

@app.command()
def start(repo_path: str):
    print(f"Start called with: {repo_path}")

if __name__ == "__main__":
    print(f"Args: {sys.argv}")
    app()