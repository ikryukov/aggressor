"""Build management functionality for HPC middleware."""

import asyncio
import logging
import os
import shlex
from pathlib import Path

from .config import BuildConfig

# Configure logger
logger = logging.getLogger(__name__)


class BuildError(Exception):
    """Exception raised for build-related errors."""

    def __init__(self, message: str, stdout: str | None = None, stderr: str | None = None):
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

    async def _run_command(self, cmd: str, cwd: Path, env: dict[str, str] | None = None) -> None:
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
        install_dir = self.config.install_dir / commit_hash
        logger.info(f"Build directory: {build_dir}")
        logger.info(f"Install directory: {install_dir}")
        if build_dir.exists() and install_dir.exists():
            return build_dir, install_dir

        if build_dir.exists():
            logger.info(f"Build directory already exists: {build_dir}")
        else:
            build_dir.mkdir(parents=True)
            logger.info(f"Created build directory: {build_dir}")

        if install_dir.exists():
            logger.info(f"Install directory already exists: {install_dir}")
        else:
            install_dir.mkdir(parents=True)
            logger.info(f"Created build directory: {build_dir}")

        try:
            # If configured, perform the build inside a Docker container
            if self.config.use_docker:
                logger.info("Docker build enabled. Building inside container environment.")

                # Ensure host build and install directories exist
                if not build_dir.exists():
                    build_dir.mkdir(parents=True, exist_ok=True)
                if not install_dir.exists():
                    install_dir.mkdir(parents=True, exist_ok=True)

                # Determine image to use or build
                image_tag = self.config.docker_image
                if image_tag is None:
                    # Build image from Dockerfile
                    repo_root = Path(__file__).resolve().parents[2]
                    image_tag = f"ucc-build-env:{self.config.docker_platform.replace('/', '-')}"
                    build_args = " ".join(
                        [f"--build-arg {shlex.quote(k)}={shlex.quote(v)}" for k, v in self.config.docker_build_args.items()]
                    )
                    docker_build_cmd = (
                        f"docker buildx build --platform {shlex.quote(self.config.docker_platform)} "
                        f"-t {shlex.quote(image_tag)} -f {shlex.quote(str(self.config.dockerfile_path))} "
                        f"{build_args} --load {shlex.quote(str(repo_root))}"
                    )
                    logger.info("Building Docker image for UCC build environment")
                    await self._run_command(docker_build_cmd, cwd=repo_root)

                # Prepare environment variable pass-throughs
                env_flags = " ".join(
                    [f"-e {shlex.quote(k)}={shlex.quote(v)}" for k, v in self.config.env_vars.items()]
                )

                # Configure and make flags (allow env var expansion like $HPCX_UCX_DIR)
                configure_flags = " ".join(flag for flag in self.config.configure_flags)
                make_flags = " ".join(flag for flag in self.config.make_flags if flag)

                # Build script executed inside container
                inner_script_lines = [
                    "set -eo pipefail",
                    "set +u",
                    "source /opt/hpcx/hpcx-init.sh && hpcx_load",
                    # Provide sane defaults if HPC-X doesn't export these variables
                    "export HPCX_DIR=${HPCX_DIR:-/opt/hpcx}",
                    "export HPCX_UCX_DIR=${HPCX_UCX_DIR:-$HPCX_DIR/ucx}",
                    "export HPCX_MPI_DIR=${HPCX_MPI_DIR:-$HPCX_DIR/ompi}",
                    "set -u",
                    "mkdir -p /workspace/build /workspace/install",
                    # Run autogen in the source tree (it writes to source)
                    "if [ -x /workspace/src/autogen.sh ]; then (cd /workspace/src && ./autogen.sh); fi",
                    "cd /workspace/build",
                    f"/workspace/src/configure --prefix=/workspace/install {configure_flags}",
                    f"make -j$(nproc) {make_flags}".rstrip(),
                    "make install",
                ]
                inner_script = "; ".join(inner_script_lines)

                docker_run_cmd = (
                    f"docker run --rm --platform {shlex.quote(self.config.docker_platform)} "
                    f"--user $(id -u):$(id -g) "
                    f"-v {shlex.quote(str(source_dir))}:/workspace/src "
                    f"-v {shlex.quote(str(build_dir))}:/workspace/build "
                    f"-v {shlex.quote(str(install_dir))}:/workspace/install "
                    f"{env_flags} "
                    f"{shlex.quote(image_tag)} bash -lc {shlex.quote(inner_script)}"
                )

                await self._run_command(docker_run_cmd, cwd=build_dir)

                logger.info(f"Docker build completed successfully for commit {commit_hash}")
                return build_dir, install_dir

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
                f"--prefix={install_dir}",
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
            return build_dir, install_dir

        except BuildError as e:
            logger.error(f"Build failed for commit {commit_hash}")
            if e.stdout:
                logger.error(f"Build stdout:\n{e.stdout}")
            if e.stderr:
                logger.error(f"Build stderr:\n{e.stderr}")
            # shutil.rmtree(build_dir, ignore_errors=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error during build: {e!s}", exc_info=True)
            # shutil.rmtree(build_dir, ignore_errors=True)
            raise BuildError(f"Unexpected error during build: {e!s}")

    async def cleanup(self) -> None:
        """Clean up build directories."""
        if self.config.build_dir.exists():
            logger.info(f"Cleaning up build directory: {self.config.build_dir}")
            # shutil.rmtree(self.config.build_dir, ignore_errors=True)