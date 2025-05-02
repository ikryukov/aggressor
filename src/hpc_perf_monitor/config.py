"""Configuration models for HPC Performance Monitor."""

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    """Supported memory types for benchmarks."""
    HOST = "host"
    CUDA = "cuda"
    ROCM = "rocm"


class BenchmarkType(str, Enum):
    """Types of benchmarks supported by the system."""
    UCC = "ucc"
    OSU = "osu"


class BuildConfig(BaseModel):
    """Configuration for building HPC middleware."""
    source_dir: Path = Field(..., description="Directory containing source code")
    build_dir: Path = Field(..., description="Directory for build artifacts")
    install_dir: Path = Field(..., description="Directory for installed artifacts")
    configure_flags: list[str] = Field(default_factory=list, description="Custom configure flags")
    make_flags: list[str] = Field(default_factory=list, description="Custom make flags")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables for build")


class BenchmarkParams(BaseModel):
    """Parameters for benchmark execution."""
    num_processes: list[int] = Field(..., description="Number of processes to test")
    procs_per_node: list[int] = Field(..., description="Processes per node to test")
    memory_types: list[MemoryType] = Field(
        default=[MemoryType.HOST],
        description="Memory types to test"
    )


class SlurmConfig(BaseModel):
    """Slurm job submission configuration."""
    partition: str = Field(..., description="Slurm partition to use")
    time_limit: str = Field(default="01:00:00", description="Job time limit (HH:MM:SS)")
    output_dir: str = Field(default="~/results", description="Directory for job output files")
    job_name: str = Field(default="ucc_benchmark", description="Name of the job")
    account: str | None = Field(None, description="Slurm account to charge")
    qos: str | None = Field(None, description="Quality of Service")
    additional_flags: dict[str, str] = Field(
        default_factory=dict,
        description="Additional Slurm flags"
    )


class BenchmarkConfig(BaseModel):
    """Configuration for a specific benchmark."""
    name: str = Field(..., description="Benchmark name")
    type: BenchmarkType = Field(..., description="Type of benchmark (ucc or osu)")
    benchmark_dir: Path = Field(..., description="Directory containing benchmark binaries")
    command: str = Field(..., description="Command to run the benchmark")
    mpi_args: str = Field(default="", description="Additional MPI arguments")
    params: BenchmarkParams = Field(..., description="Benchmark parameters")
    parser: str = Field(..., description="Name of parser plugin to use")
    metrics: list[str] = Field(..., description="Metrics to collect")
    slurm: SlurmConfig | None = Field(None, description="Slurm configuration if using Slurm")


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
    metrics_weight: dict[str, float] = Field(
        default_factory=dict,
        description="Weight of each metric in regression analysis"
    )


class ProjectConfig(BaseModel):
    """Main project configuration."""
    repo_url: str = Field(..., description="Git repository URL")
    work_dir: Path = Field(..., description="Working directory for all operations")
    ref_commit: str = Field(..., description="Reference commit to compare against")
    test_commits: list[str] = Field(..., description="Commits to test")
    build: BuildConfig = Field(..., description="Build configuration")
    benchmarks: list[BenchmarkConfig] = Field(..., description="Benchmark configurations")
    regression: RegressionConfig = Field(
        default_factory=RegressionConfig,
        description="Regression detection configuration"
    )
    report_formats: list[str] = Field(
        default=["markdown"],
        description="Output report formats (markdown, json, html). Can specify multiple formats."
    ) 