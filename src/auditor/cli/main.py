"""CLI entrypoint for dataset-auditor scaffolding."""

from __future__ import annotations

import typer

from auditor import __version__
from auditor.config import get_settings
from auditor.logging_config import configure_logging, get_logger

app = typer.Typer(
    name="auditor",
    help="Evidence-grounded dataset auditor.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Evidence-grounded dataset auditor."""


@app.command("version")
def version_cmd() -> None:
    """Print the package version and confirm configuration loads."""
    settings = get_settings()
    configure_logging(settings.log_level)
    log = get_logger("auditor.cli").bind(run_id="cli-version")
    log.info("version_check", version=__version__, log_level=settings.log_level)
    typer.echo(f"dataset-auditor {__version__}")
    typer.echo("config loaded successfully")


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":
    main()
