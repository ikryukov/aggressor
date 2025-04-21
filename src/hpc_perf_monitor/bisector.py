"""Git bisect functionality for performance analysis."""

import asyncio
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from .analyzer import load_config, run_benchmarks
from .build_manager import BuildManager
from .git_manager import GitManager
from .metrics_analyzer import MetricsAnalyzer
from .report_generator import ReportGenerator

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger("hpc_perf_monitor")


def run_bisect(
    config_file: Path = typer.Argument(..., help="Path to configuration YAML file"),
    good_commit: str = typer.Argument(..., help="Known good commit hash or reference"),
    bad_commit: str = typer.Argument(..., help="Known bad commit hash or reference"),
    output_dir: Path = typer.Option(
        Path("results"),
        help="Directory to store results"
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug logging"
    )
):
    """Perform git bisect based on performance analysis."""
    if debug:
        logger.setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(config_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize managers
    git_manager = GitManager(config.repo_url, config.work_dir)
    build_manager = BuildManager(config.build)
    metrics_analyzer = MetricsAnalyzer(config.regression)
    report_generator = ReportGenerator()

    async def main():
        try:
            # Validate and get commit information
            commit_info_dict = await git_manager.validate_commits([good_commit, bad_commit])
            good_info = commit_info_dict[good_commit]
            bad_info = commit_info_dict[bad_commit]
            
            logger.info("Starting bisect between:")
            logger.info(f"Good commit: {good_info.short_hash} ({good_info.date.strftime('%Y-%m-%d')}, {good_info.message[:30]}...)")
            logger.info(f"Bad commit: {bad_info.short_hash} ({bad_info.date.strftime('%Y-%m-%d')}, {bad_info.message[:30]}...)")
            
            # Get commit range for bisect
            commits = await git_manager.get_commits_between(good_commit, bad_commit)
            if not commits:
                logger.error("No commits found between good and bad commits")
                raise typer.Exit(1)
                
            logger.info(f"Found {len(commits)} commits to test")
            
            # Run benchmarks for good commit
            logger.info(f"Running benchmarks for good commit {good_commit}")
            good_results = await run_benchmarks(
                git_manager,
                build_manager,
                config,
                good_commit
            )
            
            # Initialize bisect state
            left = 0
            right = len(commits) - 1
            first_bad = None
            
            while left <= right:
                mid = (left + right) // 2
                current_commit = commits[mid]
                
                logger.info(f"\nTesting commit {current_commit.short_hash} ({current_commit.date.strftime('%Y-%m-%d')}, {current_commit.message[:30]}...)")
                
                # Run benchmarks for current commit
                current_results = await run_benchmarks(
                    git_manager,
                    build_manager,
                    config,
                    current_commit.hash
                )              
                
                # Analyze results
                analysis = metrics_analyzer.analyze_results(good_results, current_results)
                
                # Check if current commit is bad
                is_bad = any(result.has_regression for result in analysis)
                
                # Generate report
                for report_format in config.report_formats:
                    report_file = output_dir / f"bisect_{current_commit.short_hash}.{report_format}"
                    report_generator.generate_report(
                        analysis,
                        report_file,
                        report_format,
                        {
                            "good_commit": {
                                "hash": good_info.hash,
                                "short_hash": good_info.short_hash,
                                "author": good_info.author,
                                "date": good_info.date.strftime('%Y-%m-%d %H:%M'),
                                "message": good_info.message
                            },
                            "test_commit": {
                                "hash": current_commit.hash,
                                "short_hash": current_commit.short_hash,
                                "author": current_commit.author,
                                "date": current_commit.date.strftime('%Y-%m-%d %H:%M'),
                                "message": current_commit.message
                            }
                        }
                    )
                    logger.info(f"Generated {report_format} report: {report_file}")
                
                if is_bad:
                    logger.info(f"Commit {current_commit.short_hash} is BAD")
                    first_bad = current_commit
                    right = mid - 1
                else:
                    logger.info(f"Commit {current_commit.short_hash} is GOOD")
                    left = mid + 1
            
            if first_bad:
                logger.info("\nFirst bad commit found:")
                logger.info(f"Hash: {first_bad.hash}")
                logger.info(f"Author: {first_bad.author}")
                logger.info(f"Date: {first_bad.date.strftime('%Y-%m-%d %H:%M')}")
                logger.info(f"Message: {first_bad.message}")
            else:
                logger.info("\nNo bad commits found in the range")

        finally:
            logger.info("Cleaning up")
            # Cleanup
            # await git_manager.cleanup()
            # await build_manager.cleanup()

    # Run bisect
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Bisect failed: {e}")
        raise typer.Exit(1) 