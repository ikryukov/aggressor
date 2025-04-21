"""Benchmark execution management."""

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Tuple, Union, cast

from jinja2 import Environment, FileSystemLoader

from .config import BenchmarkConfig, MemoryType, SlurmConfig
from .parsers import get_parser


logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    commit_hash: str
    benchmark_name: str
    parameters: Dict[str, Union[int, str]]
    metrics: Dict[str, float]
    metrics_watch: List[str]
    timestamp: datetime
    stdout: str
    stderr: str


class MPICommandGenerator(Protocol):
    """Protocol for MPI command generators."""
    
    def __call__(
        self, 
        benchmark_cmd: str,
        num_processes: int,
        procs_per_node: int,
        memory_type: MemoryType,
        build_dir: Path
    ) -> str:
        """Generate MPI command.
        
        Args:
            benchmark_cmd: Command to run
            num_processes: Total number of processes
            procs_per_node: Processes per node
            memory_type: Memory type to use
            build_dir: Directory containing built binaries
            
        Returns:
            Complete MPI command as a string
        """
        ...


class BenchmarkType(Enum):
    """Types of benchmarks supported by the runner."""
    UCC_PERFTEST = "ucc_perftest"
    OSU = "osu"
    DEFAULT = "default"


class BenchmarkRunner:
    """Manages benchmark execution."""

    def __init__(self, build_dir: Path) -> None:
        """Initialize BenchmarkRunner.

        Args:
            build_dir: Directory containing built binaries
        """
        self.build_dir = build_dir
        self.env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))
        
        # Register command generators for different benchmark types
        self._command_generators: Dict[BenchmarkType, MPICommandGenerator] = {
            BenchmarkType.UCC_PERFTEST: self._generate_ucc_perftest_command,
            BenchmarkType.OSU: self._generate_osu_command,
            BenchmarkType.DEFAULT: self._generate_default_mpi_command
        }

    def _detect_benchmark_type(self, benchmark_cmd: str) -> BenchmarkType:
        """Detect benchmark type based on command string.
        
        Args:
            benchmark_cmd: Benchmark command to analyze
            
        Returns:
            Detected benchmark type
        """
        if benchmark_cmd.startswith("ucc_perftest"):
            return BenchmarkType.UCC_PERFTEST
        elif benchmark_cmd.startswith("osu_"):
            return BenchmarkType.OSU
        else:
            return BenchmarkType.DEFAULT

    def _generate_mpi_hostfile(self, num_nodes: int) -> Path:
        """Generate MPI hostfile.

        Args:
            num_nodes: Number of nodes required

        Returns:
            Path to generated hostfile
        """
        hostfile = Path(tempfile.mktemp(prefix="hostfile_"))
        with hostfile.open("w") as f:
            # In a real cluster, this would be populated with actual node names
            for i in range(num_nodes):
                f.write(f"node{i} slots=unlimited\n")
        return hostfile

    def _generate_mpi_command(
        self,
        benchmark_cmd: str,
        num_processes: int,
        procs_per_node: int,
        memory_type: MemoryType,
        bench_dir: Path
    ) -> str:
        """Generate MPI run command based on benchmark type.

        Args:
            benchmark_cmd: Benchmark command to run
            num_processes: Total number of processes
            procs_per_node: Processes per node
            memory_type: Memory type to use

        Returns:
            Complete MPI command
        """
        benchmark_type = self._detect_benchmark_type(benchmark_cmd)
        generator = self._command_generators[benchmark_type]
        
        return generator(
            benchmark_cmd,
            num_processes,
            procs_per_node,
            memory_type,
            self.build_dir,
            bench_dir
        )

    def _generate_ucc_perftest_command(
        self,
        benchmark_cmd: str,
        num_processes: int,
        procs_per_node: int,
        memory_type: MemoryType,
        build_dir: Path,
        bench_dir: Path
    ) -> str:
        """Generate MPI command specifically for UCC perftest.

        Args:
            benchmark_cmd: UCC perftest command to run
            num_processes: Total number of processes
            procs_per_node: Processes per node
            memory_type: Memory type to use
            build_dir: Directory containing built binaries

        Returns:
            Complete MPI command for UCC perftest
        """
        num_nodes = (num_processes + procs_per_node - 1) // procs_per_node
        
        mpi_cmd = [
            "mpirun",
            "-np", str(num_processes),
            "-x", f"LD_LIBRARY_PATH={build_dir}/lib:$LD_LIBRARY_PATH"
        ]

        # UCC-specific flags
        mpi_cmd.extend([
            "--mca", "coll", "^hcoll", 
            "--mca", "coll_ucc_enable", "0"
        ])

        # Memory type specific flags for UCC
        if memory_type == MemoryType.CUDA:
            mpi_cmd.extend(["-x", "UCC_TLS=cuda,ucp"])
            
        elif memory_type == MemoryType.ROCM:
            mpi_cmd.extend(["-x", "UCC_TLS=rocm,ucp"])

        mpi_cmd.append(f"{build_dir}/bin/{benchmark_cmd}")

        if memory_type == MemoryType.CUDA:
            mpi_cmd.extend(["-m", "cuda"])
        
        # mpi_cmd.extend(["-n", "10000"]) 

        return " ".join(mpi_cmd)

    def _generate_osu_command(
        self,
        benchmark_cmd: str,
        num_processes: int,
        procs_per_node: int,
        memory_type: MemoryType,
        build_dir: Path,
        bench_dir: Path
    ) -> str:
        """Generate MPI command specifically for OSU benchmarks.

        Args:
            benchmark_cmd: OSU benchmark command to run
            num_processes: Total number of processes
            procs_per_node: Processes per node
            memory_type: Memory type to use
            build_dir: Directory containing built binaries
        Returns:
            Complete MPI command for OSU benchmark
        """
        num_nodes = (num_processes + procs_per_node - 1) // procs_per_node
        
        mpi_cmd = [
            "mpirun",
            "-np", str(num_processes),
            "-x", f"LD_LIBRARY_PATH={build_dir}/lib:$LD_LIBRARY_PATH"            
        ]

        # UCC-specific flags
        mpi_cmd.extend([
            "--mca", "coll", "^hcoll", 
            "--mca", "coll_ucc_enable", "0"
        ])

        # Add mapping by node for OSU benchmarks
        # if num_nodes > 1:
        #     mpi_cmd.extend([
        #         "--map-by", f"ppr:{procs_per_node}:node"
        #     ])


        mpi_cmd.append(f"{bench_dir}/{benchmark_cmd}")
        
        return " ".join(mpi_cmd)

    def _generate_default_mpi_command(
        self,
        benchmark_cmd: str,
        num_processes: int,
        procs_per_node: int,
        memory_type: MemoryType,
        build_dir: Path
    ) -> str:
        """Generate default MPI command for other benchmarks.

        Args:
            benchmark_cmd: Benchmark command to run
            num_processes: Total number of processes
            procs_per_node: Processes per node
            memory_type: Memory type to use
            build_dir: Directory containing built binaries

        Returns:
            Complete MPI command
        """
        num_nodes = (num_processes + procs_per_node - 1) // procs_per_node
        
        mpi_cmd = [
            "mpirun",
            "-np", str(num_processes),
            "-x", f"LD_LIBRARY_PATH={build_dir}/lib:$LD_LIBRARY_PATH",
            "--tag-output"
        ]

        # Add memory type specific flags for generic benchmarks
        if memory_type == MemoryType.CUDA:
            mpi_cmd.extend(["-x", "CUDA_VISIBLE_DEVICES=0,1,2,3"])
        elif memory_type == MemoryType.ROCM:
            mpi_cmd.extend(["-x", "ROCR_VISIBLE_DEVICES=0,1,2,3"])

        mpi_cmd.append(f"{build_dir}/bin/{benchmark_cmd}")
        
        return " ".join(mpi_cmd)

    def _generate_slurm_script(
        self,
        config: SlurmConfig,
        mpi_cmd: str,
        num_nodes: int
    ) -> str:
        """Generate Slurm batch script.

        Args:
            config: Slurm configuration
            mpi_cmd: MPI command to run
            num_nodes: Number of nodes required

        Returns:
            Slurm batch script content
        """
        template = self.env.get_template("slurm_job.sh.j2")
        return template.render(
            partition=config.partition,
            time_limit=config.time_limit,
            num_nodes=num_nodes,
            account=config.account,
            qos=config.qos,
            additional_flags=config.additional_flags,
            mpi_cmd=mpi_cmd
        )

    async def _run_local(self, cmd: str) -> Tuple[str, str, int]:
        """Run command locally.

        Args:
            cmd: Command to run

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        process = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        return stdout.decode(), stderr.decode(), process.returncode

    async def _run_slurm(
        self,
        script_content: str,
        job_name: str
    ) -> Tuple[str, str, int]:
        """Submit and monitor Slurm job.

        Args:
            script_content: Slurm batch script content
            job_name: Name for the Slurm job

        Returns:
            Tuple of (stdout, stderr, return_code)
        """
        # Write script to temporary file
        script_file = Path(tempfile.mktemp(prefix=f"slurm_{job_name}_"))
        script_file.write_text(script_content)

        try:
            # Submit job
            proc = await asyncio.create_subprocess_shell(
                f"sbatch --parsable {script_file}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                return "", stderr.decode(), proc.returncode

            job_id = stdout.decode().strip()

            # Wait for job completion
            while True:
                proc = await asyncio.create_subprocess_shell(
                    f"scontrol show job {job_id}",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()
                if "JobState=COMPLETED" in stdout.decode():
                    break
                elif any(state in stdout.decode() for state in ["FAILED", "CANCELLED", "TIMEOUT"]):
                    return "", f"Job failed with state: {stdout.decode()}", 1
                await asyncio.sleep(10)

            # Get output
            with open(f"slurm-{job_id}.out") as f:
                stdout = f.read()
            with open(f"slurm-{job_id}.err") as f:
                stderr = f.read()

            return stdout, stderr, 0

        finally:
            script_file.unlink()

    async def run_benchmark(
        self,
        commit_hash: str,
        config: BenchmarkConfig,
        params: Dict[str, Union[int, str]]
    ) -> BenchmarkResult:
        """Run a benchmark with specific parameters.

        Args:
            commit_hash: Git commit hash being tested
            config: Benchmark configuration
            params: Benchmark parameters

        Returns:
            Benchmark results

        Raises:
            RuntimeError: If benchmark execution fails
        """
        num_processes = cast(int, params["num_processes"])
        procs_per_node = cast(int, params["procs_per_node"])
        memory_type = MemoryType(params["memory_type"])
        bench_dir = config.benchmark_dir

        # Generate MPI command
        mpi_cmd = self._generate_mpi_command(
            config.command,
            num_processes,
            procs_per_node,
            memory_type,
            bench_dir
        )
                
        logger.info(f"MPI command: {mpi_cmd}")

        # Run benchmark
        if config.slurm:
            num_nodes = (num_processes + procs_per_node - 1) // procs_per_node
            script = self._generate_slurm_script(config.slurm, mpi_cmd, num_nodes)
            stdout, stderr, rc = await self._run_slurm(
                script,
                f"{config.name}_{commit_hash}"
            )
        else:
            stdout, stderr, rc = await self._run_local(mpi_cmd)

        if rc != 0:
            raise RuntimeError(
                f"Benchmark failed with exit code {rc}\nstdout: {stdout}\nstderr: {stderr}"
            )

        # Parse results using the specified parser
        parser = get_parser(config.parser)
        metrics = parser.parse(stdout, stderr)

        logger.debug("Parsed metrics: %s", metrics)

        return BenchmarkResult(
            commit_hash=commit_hash,
            benchmark_name=config.name,
            parameters=params,
            metrics=metrics,
            metrics_watch=config.metrics,
            timestamp=datetime.now(),
            stdout=stdout,
            stderr=stderr
        ) 