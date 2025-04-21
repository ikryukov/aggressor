"""Command-line interface for HPC Performance Monitor."""

import asyncio
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import typer
import yaml
from rich.console import Console
from rich.logging import RichHandler

from .benchmark_runner import BenchmarkRunner
from .build_manager import BuildManager
from .config import ProjectConfig
from .git_manager import GitManager, CommitInfo
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


async def run_benchmarks(
    git_manager: GitManager,
    build_manager: BuildManager,
    config: ProjectConfig,
    commit_hash: str
) -> List[dict]:
    """Run benchmarks for a specific commit.

    Args:
        git_manager: GitManager instance
        build_manager: BuildManager instance
        config: Project configuration
        commit_hash: Git commit hash to test

    Returns:
        List of benchmark results
    """
    # Prepare source code
    source_dir = await git_manager.prepare_commit(commit_hash, config.build)
    logger.info(f"Prepared source code for commit {commit_hash}")

    # Build
    build_dir = await build_manager.build(source_dir, commit_hash)
    logger.info(f"Built commit {commit_hash}")

    # Run benchmarks
    runner = BenchmarkRunner(build_dir)
    results = []

    for benchmark in config.benchmarks:
        logger.info(f"Running benchmark {benchmark.name}")
        
        # Generate parameter combinations
        params_list = []
        for np in benchmark.params.num_processes:
            for ppn in benchmark.params.procs_per_node:
                for mem in benchmark.params.memory_types:
                    params_list.append({
                        "num_processes": np,
                        "procs_per_node": ppn,
                        "memory_type": mem,
                    })

        # Run each parameter combination
        for params in params_list:
            try:
                result = await runner.run_benchmark(commit_hash, benchmark, params)
                results.append(result)
            except Exception as e:
                logger.error(f"Benchmark failed: {e}")

    return results


def run_analysis(
    config_file: Path = typer.Argument(..., help="Path to configuration YAML file"),
    output_dir: Path = typer.Option(
        Path("results"),
        help="Directory to store results"
    ),
    debug: bool = typer.Option(
        False,
        help="Enable debug logging"
    )
):
    """Analyze performance between commits."""
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
            commit_info_dict = await git_manager.validate_commits(
                [config.ref_commit] + config.test_commits
            )
            logger.info(f"Validated {len(commit_info_dict)} commits")
            
            # Log detailed commit information
            for commit_hash, info in commit_info_dict.items():
                logger.info(
                    f"Commit: {info.short_hash} | "
                    f"Author: {info.author} | "
                    f"Date: {info.date.strftime('%Y-%m-%d %H:%M')} | "
                    f"Message: {info.message[:50]}{'...' if len(info.message) > 50 else ''}"
                )
            
            # Create a mapping of input references to resolved hashes
            input_commits = [config.ref_commit] + config.test_commits
            logger.info(f"Resolving commit references: {input_commits}")
            ref_to_hash = {}
            
            # Clone a temporary repo to resolve references
            temp_dir = config.work_dir / "temp_ref_resolve"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                
            try:
                from git import Repo
                repo = Repo.clone_from(config.repo_url, temp_dir)
                
                # Resolve each reference to its hash
                for commit_ref in input_commits:
                    try:
                        commit = repo.commit(commit_ref)
                        ref_to_hash[commit_ref] = commit.hexsha
                        logger.info(f"Resolved '{commit_ref}' to {commit.hexsha[:8]}")
                    except Exception as e:
                        logger.error(f"Failed to resolve reference '{commit_ref}': {e}")
            finally:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Create an ordered list of commit hashes to process
            commit_hashes = []
            for commit_ref in input_commits:
                if commit_ref in ref_to_hash:
                    full_hash = ref_to_hash[commit_ref]
                    if full_hash in commit_info_dict:
                        commit_hashes.append(full_hash)
                    else:
                        logger.warning(
                            f"Resolved hash {full_hash[:8]} for '{commit_ref}' not found in validated commits. "
                            f"This may happen if the reference points to a commit not yet fetched."
                        )
                else:
                    logger.warning(
                        f"Could not resolve reference '{commit_ref}'. "
                        f"Check that it exists in the repository and is accessible."
                    )
            
            # Make sure we have a reference commit
            if not commit_hashes:
                logger.error("No valid commits found")
                return
            
            # Run reference benchmarks
            ref_hash = commit_hashes[0]
            ref_info = commit_info_dict[ref_hash]
            logger.info(f"Running reference benchmarks for {ref_info.short_hash} ({ref_info.message[:30]}...)")
            
            ref_results = await run_benchmarks(
                git_manager,
                build_manager,
                config,
                ref_hash
            )

            # Run test benchmarks and analyze
            for test_hash in commit_hashes[1:]:
                test_info = commit_info_dict[test_hash]
                logger.info(
                    f"Running benchmarks for {test_info.short_hash} "
                    f"({test_info.date.strftime('%Y-%m-%d')}, {test_info.message[:30]}...)"
                )
                
                test_results = await run_benchmarks(
                    git_manager,
                    build_manager,
                    config,
                    test_hash
                )

                logger.info(f"Test results: {test_results}")

                # Display result summary
                logger.info(f"Reference commit: {ref_info.short_hash}")
                logger.info(f"Test commit: {test_info.short_hash}")

                # Analyze results
                analysis = metrics_analyzer.analyze_results(ref_results, test_results)
                logger.info(f"Analysis results: {analysis}")

                # Generate reports in all requested formats
                for report_format in config.report_formats:
                    report_file = output_dir / f"report_{test_info.short_hash}.{report_format}"
                    report_generator.generate_report(
                        analysis,
                        report_file,
                        report_format,
                        {
                            "ref_commit": {
                                "hash": ref_info.hash,
                                "short_hash": ref_info.short_hash,
                                "author": ref_info.author,
                                "date": ref_info.date.strftime('%Y-%m-%d %H:%M'),
                                "message": ref_info.message
                            },
                            "test_commit": {
                                "hash": test_info.hash, 
                                "short_hash": test_info.short_hash,
                                "author": test_info.author,
                                "date": test_info.date.strftime('%Y-%m-%d %H:%M'),
                                "message": test_info.message
                            }
                        }
                    )
                    logger.info(f"Generated {report_format} report: {report_file}")

        finally:
            logger.info("Cleaning up")
            # Cleanup
            # await git_manager.cleanup()
            # await build_manager.cleanup()

    # Run analysis
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise typer.Exit(1)


app = typer.Typer()
app.command()(run_analysis)

def main():
    """Entry point for the CLI."""
    app() 