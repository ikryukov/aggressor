"""Command-line interface for HPC Performance Monitor."""

import asyncio
import logging
import os
from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

from .ai_diff_analyzer import AIDiffAnalysisConfig, AIDiffAnalyzer
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


def run_ai_diff_analysis(
    repo_path: Path = typer.Argument(..., help="Path to the git repository"),
    commit: str = typer.Option(..., help="Commit hash to analyze"),
    api_key: str = typer.Option(None, help="OpenAI API key"),
    api_base_url: str = typer.Option(
        "https://integrate.api.nvidia.com/v1", 
        help="OpenAI API base URL"
    ),
    model: str = typer.Option(
        "nvdev/meta/llama-3.1-70b-instruct", 
        help="AI model to use for analysis"
    ),
    output_file: Path = typer.Option(
        None, 
        help="File to save analysis results (use {commit_hash} as a placeholder)"
    ),
    system_prompt: str = typer.Option(
        None,
        help="Custom system prompt for the AI analysis"
    ),
    benchmark_name: str = typer.Option(
        None,
        help="Benchmark name for additional context"
    ),
    regression_details: str = typer.Option(
        None,
        help="Details about performance regression for additional context"
    )
):
    """Analyze a git diff using AI models to identify potential performance issues."""
    # Get API key from environment variable if not provided
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            logger.error("No API key provided. Set it with --api-key or OPENAI_API_KEY environment variable")
            raise typer.Exit(1)
    
    # Create configuration
    config = AIDiffAnalysisConfig(
        api_base_url=api_base_url,
        api_key=api_key,
        model=model,
        output_file=output_file
    )
    
    # Set custom system prompt if provided
    if system_prompt:
        config.system_prompt = system_prompt
    
    # Create analyzer
    analyzer = AIDiffAnalyzer(config)

    # Prepare benchmark info if provided
    benchmark_info = None
    if benchmark_name or regression_details:
        benchmark_info = {}
        if benchmark_name:
            benchmark_info["benchmark_name"] = benchmark_name
        if regression_details:
            benchmark_info["regression_details"] = regression_details
    
    async def _run_analysis():
        try:
            # Analyze the commit
            logger.info(f"Analyzing diff for commit {commit}")
            result = await analyzer.analyze_commit(repo_path, commit, benchmark_info)
            
            # Print analysis to console
            console.print("\n[bold blue]Analysis:[/bold blue]")
            console.print(result.analysis)
            
            return 0
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return 1
    
    # Run the analysis
    return asyncio.run(_run_analysis())


app = typer.Typer()
app.command()(run_analysis)
app.command()(run_bisect)
app.command("ai-diff")(run_ai_diff_analysis)

def main():
    """Entry point for the CLI."""
    app() 