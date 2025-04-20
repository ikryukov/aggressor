"""Git repository management functionality."""

import asyncio
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from git import Repo
from git.exc import GitCommandError

from .config import BuildConfig


@dataclass
class CommitInfo:
    """Information about a git commit."""
    
    hash: str
    short_hash: str
    author: str
    author_email: str
    date: datetime
    message: str


class GitManager:
    """Manages Git repository operations."""

    def __init__(self, repo_url: str, work_dir: Path):
        """Initialize GitManager.

        Args:
            repo_url: URL of the Git repository
            work_dir: Working directory for Git operations
        """
        self.repo_url = repo_url
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def prepare_commit(self, commit_ref: str, build_config: BuildConfig) -> Path:
        """Clone repository and checkout specific commit.

        Args:
            commit_ref: Git commit hash or reference to checkout
            build_config: Build configuration

        Returns:
            Path to the prepared source directory

        Raises:
            GitCommandError: If Git operations fail
        """
        # First resolve the reference to get the exact commit hash
        temp_dir = self.work_dir / "temp_resolve"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            
        try:
            # Clone repo to resolve the reference
            temp_repo = Repo.clone_from(self.repo_url, temp_dir)
            commit_hash = temp_repo.commit(commit_ref).hexsha
        except GitCommandError as e:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise GitCommandError(f"Failed to resolve commit reference {commit_ref}", e.status, e.stderr)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        # Now checkout the resolved commit hash
        source_dir = build_config.source_dir / commit_hash
        if source_dir.exists():
            shutil.rmtree(source_dir)

        source_dir.mkdir(parents=True)

        try:
            # Clone repository
            repo = Repo.clone_from(self.repo_url, source_dir)
            
            # Checkout specific commit
            repo.git.checkout(commit_hash)
            
            return source_dir

        except GitCommandError as e:
            shutil.rmtree(source_dir, ignore_errors=True)
            raise GitCommandError(f"Failed to prepare commit {commit_hash}", e.status, e.stderr)

    async def validate_commits(self, commits: List[str]) -> Dict[str, CommitInfo]:
        """Validate that all commits exist in the repository and collect detailed information.

        Args:
            commits: List of commit hashes or references to validate

        Returns:
            Dictionary mapping commit hashes to CommitInfo objects

        Raises:
            ValueError: If any commit is invalid
        """
        temp_dir = self.work_dir / "temp_validate"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        try:
            repo = Repo.clone_from(self.repo_url, temp_dir)
            commit_info_dict = {}

            for commit_ref in commits:
                try:
                    # Try to resolve the reference (handles branches, tags, HEAD, etc.)
                    commit_obj = repo.commit(commit_ref)
                    
                    # Extract detailed information
                    info = CommitInfo(
                        hash=commit_obj.hexsha,
                        short_hash=commit_obj.hexsha[:8],
                        author=commit_obj.author.name,
                        author_email=commit_obj.author.email,
                        date=datetime.fromtimestamp(commit_obj.committed_date),
                        message=commit_obj.message.strip()
                    )
                    
                    # Store with both the original reference and the hash as keys
                    # This makes it easy to look up by either the reference or hash
                    commit_info_dict[commit_obj.hexsha] = info
                    
                except GitCommandError as e:
                    raise ValueError(f"Invalid commit reference: {commit_ref}. Error: {str(e)}")

            return commit_info_dict

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def get_commits_between(self, start: str, end: str) -> Dict[str, CommitInfo]:
        """Get detailed information about commits between two points.

        Args:
            start: Starting commit reference (hash, branch, tag, etc.)
            end: Ending commit reference (hash, branch, tag, etc.)

        Returns:
            Dictionary mapping commit hashes to CommitInfo objects

        Raises:
            GitCommandError: If Git operations fail
        """
        temp_dir = self.work_dir / "temp_range"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        try:
            repo = Repo.clone_from(self.repo_url, temp_dir)
            
            try:
                # Resolve references to full commit objects
                start_commit = repo.commit(start)
                end_commit = repo.commit(end)
                
                # Use the resolved commits with ^ to include the start commit
                commit_range = f"{start_commit.hexsha}^..{end_commit.hexsha}"
                commits = list(repo.iter_commits(commit_range))
            except GitCommandError as e:
                raise ValueError(f"Invalid commit references. Error: {str(e)}")
            
            # Create dictionary of commit info
            commit_info_dict = {}
            for commit_obj in reversed(commits):
                info = CommitInfo(
                    hash=commit_obj.hexsha,
                    short_hash=commit_obj.hexsha[:8],
                    author=commit_obj.author.name,
                    author_email=commit_obj.author.email,
                    date=datetime.fromtimestamp(commit_obj.committed_date),
                    message=commit_obj.message.strip()
                )
                commit_info_dict[commit_obj.hexsha] = info
                
            return commit_info_dict

        except GitCommandError as e:
            raise GitCommandError(f"Failed to get commits between {start} and {end}", 
                                e.status, e.stderr)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def get_commit_range_info(self, start: str, end: str) -> Dict[str, CommitInfo]:
        """Get detailed information about a range of commits.
        
        This function gets all commits between start and end (inclusive of both),
        and returns detailed information about each commit.

        Args:
            start: Starting commit reference (hash, branch, tag, etc.)
            end: Ending commit reference (hash, branch, tag, etc.)

        Returns:
            Dictionary mapping commit hashes to CommitInfo objects

        Raises:
            GitCommandError: If Git operations fail
            ValueError: If start or end commit reference is invalid
        """
        temp_dir = self.work_dir / "temp_range_info"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        try:
            repo = Repo.clone_from(self.repo_url, temp_dir)
            
            try:
                # Resolve references to full commit objects
                start_commit = repo.commit(start)
                end_commit = repo.commit(end)
                
                # Get the full commit range
                commit_range = f"{start_commit.hexsha}^..{end_commit.hexsha}"
                commits = list(repo.iter_commits(commit_range))
            except GitCommandError as e:
                raise ValueError(f"Invalid commit reference: {start} or {end}. Error: {str(e)}")
            
            # Create dictionary of commit info
            commit_info_dict = {}
            for commit_obj in reversed(commits):
                info = CommitInfo(
                    hash=commit_obj.hexsha,
                    short_hash=commit_obj.hexsha[:8],
                    author=commit_obj.author.name,
                    author_email=commit_obj.author.email,
                    date=datetime.fromtimestamp(commit_obj.committed_date),
                    message=commit_obj.message.strip()
                )
                commit_info_dict[commit_obj.hexsha] = info
                
            return commit_info_dict

        except GitCommandError as e:
            raise GitCommandError(f"Failed to get commits between {start} and {end}", 
                                e.status, e.stderr)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def get_commit_info(self, commit_ref: str) -> CommitInfo:
        """Get detailed information about a single commit.

        Args:
            commit_ref: Commit hash or reference (branch, tag, HEAD, etc.)

        Returns:
            CommitInfo object with commit details

        Raises:
            GitCommandError: If Git operations fail
            ValueError: If commit reference is invalid
        """
        temp_dir = self.work_dir / "temp_commit_info"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        try:
            repo = Repo.clone_from(self.repo_url, temp_dir)
            
            try:
                # Resolve the reference
                commit_obj = repo.commit(commit_ref)
                
                # Extract commit details
                info = CommitInfo(
                    hash=commit_obj.hexsha,
                    short_hash=commit_obj.hexsha[:8],
                    author=commit_obj.author.name,
                    author_email=commit_obj.author.email,
                    date=datetime.fromtimestamp(commit_obj.committed_date),
                    message=commit_obj.message.strip()
                )
                
                return info
                
            except GitCommandError as e:
                raise ValueError(f"Invalid commit reference: {commit_ref}. Error: {str(e)}")

        except GitCommandError as e:
            raise GitCommandError(f"Failed to get info for commit {commit_ref}", 
                                 e.status, e.stderr)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    async def cleanup(self) -> None:
        """Clean up temporary directories."""
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir, ignore_errors=True) 