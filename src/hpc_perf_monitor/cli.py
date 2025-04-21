"""Command-line interface for HPC Performance Monitor."""

import logging
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

from .analyzer import run_analysis
from .bisector import run_bisect
from .config import ProjectConfig

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger("hpc_perf_monitor")


def load_config(config_file: Path) -> ProjectConfig:
    """Load configuration from YAML file.

    Args:
        config_file: Path to configuration file

    Returns:
        ProjectConfig object

    Raises:
        typer.Exit: If configuration is invalid
    """
    try:
        with config_file.open() as f:
            config_dict = yaml.safe_load(f)
        return ProjectConfig(**config_dict)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise typer.Exit(1)


app = typer.Typer()
app.command()(run_analysis)
app.command()(run_bisect)

def main():
    """Entry point for the CLI."""
    app() 