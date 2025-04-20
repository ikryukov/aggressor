"""Configuration models for HPC Performance Monitor."""

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Supported memory types for benchmarks."""
    HOST = "host"
    CUDA = "cuda"
    ROCM = "rocm"


class BuildConfig(BaseModel):
    """Configuration for building HPC middleware."""
    source_dir: Path = Field(..., description="Directory containing source code")
    build_dir: Path = Field(..., description="Directory for build artifacts")
    configure_flags: List[str] = Field(default_factory=list, description="Custom configure flags")
    make_flags: List[str] = Field(default_factory=list, description="Custom make flags")
    env_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables for build")


class BenchmarkParams(BaseModel):
    """Parameters for benchmark execution."""
    num_processes: List[int] = Field(..., description="Number of processes to test")
    msg_sizes: List[int] = Field(..., description="Message sizes to test (in bytes)")
    procs_per_node: List[int] = Field(..., description="Processes per node to test")
    memory_types: List[MemoryType] = Field(
        default=[MemoryType.HOST],
        description="Memory types to test"
    )


class SlurmConfig(BaseModel):
    """Slurm job submission configuration."""
    partition: str = Field(..., description="Slurm partition to use")
    time_limit: str = Field(default="01:00:00", description="Job time limit (HH:MM:SS)")
    account: Optional[str] = Field(None, description="Slurm account to charge")
    qos: Optional[str] = Field(None, description="Quality of Service")
    additional_flags: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional Slurm flags"
    )


class BenchmarkConfig(BaseModel):
    """Configuration for a specific benchmark."""
    name: str = Field(..., description="Benchmark name")
    command: str = Field(..., description="Command to run the benchmark")
    params: BenchmarkParams = Field(..., description="Benchmark parameters")
    parser: str = Field(..., description="Name of parser plugin to use")
    metrics: List[str] = Field(..., description="Metrics to collect")
    slurm: Optional[SlurmConfig] = Field(None, description="Slurm configuration if using Slurm")


class RegressionConfig(BaseModel):
    """Configuration for regression detection."""
    threshold_pct: float = Field(
        default=5.0,
        description="Percentage threshold for regression detection"
    )
    min_runs: int = Field(
        default=3,
        description="Minimum number of runs for statistical significance"
    )
    metrics_weight: Dict[str, float] = Field(
        default_factory=dict,
        description="Weight of each metric in regression analysis"
    )


class ProjectConfig(BaseModel):
    """Main project configuration."""
    repo_url: str = Field(..., description="Git repository URL")
    work_dir: Path = Field(..., description="Working directory for all operations")
    ref_commit: str = Field(..., description="Reference commit to compare against")
    test_commits: List[str] = Field(..., description="Commits to test")
    build: BuildConfig = Field(..., description="Build configuration")
    benchmarks: List[BenchmarkConfig] = Field(..., description="Benchmark configurations")
    regression: RegressionConfig = Field(
        default_factory=RegressionConfig,
        description="Regression detection configuration"
    )
    report_formats: List[str] = Field(
        default=["markdown"],
        description="Output report formats (markdown, json, html). Can specify multiple formats."
    ) 