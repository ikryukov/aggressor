"""Build management functionality for HPC middleware."""

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

from .config import BuildConfig

# Configure logger
logger = logging.getLogger(__name__)


class BuildError(Exception):
    """Exception raised for build-related errors."""

    def __init__(self, message: str, stdout: Optional[str] = None, stderr: Optional[str] = None):
        """Initialize BuildError.

        Args:
            message: Error message
            stdout: Standard output from the build process
            stderr: Standard error from the build process
        """
        super().__init__(message)
        self.stdout = stdout
        self.stderr = stderr


class BuildManager:
    """Manages building of HPC middleware."""

    def __init__(self, build_config: BuildConfig):
        """Initialize BuildManager.

        Args:
            build_config: Build configuration
        """
        self.config = build_config
        self.config.build_dir.mkdir(parents=True, exist_ok=True)

    async def _run_command(self, cmd: str, cwd: Path, env: Optional[Dict[str, str]] = None) -> None:
        """Run a shell command asynchronously.

        Args:
            cmd: Command to run
            cwd: Working directory
            env: Environment variables

        Raises:
            BuildError: If command execution fails
        """
        logger.info(f"Running command: {cmd} in directory: {cwd}")
        process = await asyncio.create_subprocess_shell(
            cmd,
            cwd=cwd,
            env={**os.environ, **(env or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await process.communicate()
        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""

        if process.returncode != 0:
            logger.error(f"Command failed with exit code {process.returncode}")
            if stdout_str:
                logger.error(f"Command stdout:\n{stdout_str}")
            if stderr_str:
                logger.error(f"Command stderr:\n{stderr_str}")
            raise BuildError(
                f"Command failed with exit code {process.returncode}",
                stdout=stdout_str,
                stderr=stderr_str
            )
        logger.debug(f"Command completed successfully. Output:\n{stdout_str}")

    async def build(self, source_dir: Path, commit_hash: str) -> Path:
        """Build HPC middleware from source.

        Args:
            source_dir: Directory containing source code
            commit_hash: Git commit hash being built

        Returns:
            Path to the build directory

        Raises:
            BuildError: If build fails
        """
        logger.info(f"Starting build for commit {commit_hash}")
        build_dir = self.config.build_dir / commit_hash
        if build_dir.exists():
            return build_dir
            # logger.info(f"Removing existing build directory: {build_dir}")
            # shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)
        logger.info(f"Created build directory: {build_dir}")

        try:
            # Run autogen if present
            if (source_dir / "autogen.sh").exists():
                logger.info("Running autogen.sh")
                try:
                    await self._run_command(
                        "./autogen.sh",
                        cwd=source_dir,
                        env=self.config.env_vars
                    )
                except BuildError as e:
                    logger.error("autogen.sh step failed")
                    raise BuildError("autogen.sh step failed", stdout=e.stdout, stderr=e.stderr)

            # Configure
            logger.info("Running configure step")
            configure_cmd = [
                f"{source_dir}/configure",
                f"--prefix={build_dir}",
                *self.config.configure_flags
            ]
            try:
                await self._run_command(
                    " ".join(configure_cmd),
                    cwd=build_dir,
                    env=self.config.env_vars
                )
            except BuildError as e:
                logger.error("Configure step failed")
                raise BuildError("Configure step failed", stdout=e.stdout, stderr=e.stderr)

            # Build
            logger.info("Running build step")
            make_cmd = ["make", "-j", str(os.cpu_count() or 1)]
            if self.config.make_flags:
                make_cmd.extend(self.config.make_flags)
            try:
                await self._run_command(
                    " ".join(make_cmd),
                    cwd=build_dir,
                    env=self.config.env_vars
                )
            except BuildError as e:
                logger.error("Build step failed")
                raise BuildError("Build step failed", stdout=e.stdout, stderr=e.stderr)

            # Install
            logger.info("Running install step")
            try:
                await self._run_command(
                    "make install",
                    cwd=build_dir,
                    env=self.config.env_vars
                )
            except BuildError as e:
                logger.error("Install step failed")
                raise BuildError("Install step failed", stdout=e.stdout, stderr=e.stderr)

            logger.info(f"Build completed successfully for commit {commit_hash}")
            return build_dir

        except BuildError as e:
            logger.error(f"Build failed for commit {commit_hash}")
            if e.stdout:
                logger.error(f"Build stdout:\n{e.stdout}")
            if e.stderr:
                logger.error(f"Build stderr:\n{e.stderr}")
            # shutil.rmtree(build_dir, ignore_errors=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during build: {str(e)}", exc_info=True)
            # shutil.rmtree(build_dir, ignore_errors=True)
            raise BuildError(f"Unexpected error during build: {str(e)}")

    async def cleanup(self) -> None:
        """Clean up build directories."""
        if self.config.build_dir.exists():
            logger.info(f"Cleaning up build directory: {self.config.build_dir}")
            # shutil.rmtree(self.config.build_dir, ignore_errors=True) 