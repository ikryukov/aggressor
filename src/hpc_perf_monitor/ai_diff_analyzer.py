"""AI-based git diff analysis functionality."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Union

from git import Repo
from git.exc import GitCommandError
from openai import OpenAI
from pydantic import BaseModel

logger = logging.getLogger("hpc_perf_monitor")


class AIDiffAnalysisConfig(BaseModel):
    """Configuration for AI-based diff analysis."""
    
    # OpenAI API configuration
    api_base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    model: str = "nvdev/meta/llama-3.1-70b-instruct"
    
    # Analysis parameters
    temperature: float = 0.2
    top_p: float = 0.7
    max_tokens: int = 1024
    
    # Analysis prompts
    system_prompt: str = (
        "You are a performance analysis expert. Analyze the provided code diff and "
        "identify any performance degradation issues. Focus on computational "
        "complexity, memory usage, and bottlenecks. Provide short and concise "
        "analysis, no more than 100 words."
    )
    
    # Output configuration
    output_file: Optional[Path] = None


@dataclass
class DiffAnalysisResult:
    """Result of AI-based diff analysis."""
    
    commit_hash: str
    diff_content: str
    analysis: str


class AIDiffAnalyzer:
    """Analyzes git diffs using AI models."""

    def __init__(self, config: AIDiffAnalysisConfig):
        """Initialize the AI diff analyzer.
        
        Args:
            config: Configuration for AI-based diff analysis
        """
        self.config = config
        self.client = OpenAI(
            base_url=config.api_base_url,
            api_key=config.api_key
        )
        
    async def get_commit_diff(self, repo_path: str, commit_hash: str) -> str:
        """Get the diff for a specific commit.
        
        Args:
            repo_path: Path to the git repository
            commit_hash: Hash of the commit to analyze
            
        Returns:
            String containing the diff content
            
        Raises:
            GitCommandError: If git operations fail
        """
        try:
            repo = Repo(repo_path)
            commit = repo.commit(commit_hash)
            
            # Get the diff for this commit
            if commit.parents:
                parent = commit.parents[0]
                diff = repo.git.diff(parent.hexsha, commit.hexsha)
            else:
                # For initial commits with no parent
                diff = repo.git.show(commit.hexsha)
                
            return diff
        except GitCommandError as e:
            logger.error(f"Failed to get diff for commit {commit_hash}: {e}")
            raise
    
    async def analyze_diff(self, diff_content: str, benchmark_info: dict = None) -> str:
        """Analyze a diff using AI models.
        
        Args:
            diff_content: Content of the git diff to analyze
            benchmark_info: Optional dictionary with benchmark information to provide context
            
        Returns:
            Analysis text from the AI model
        """
        # Prepare system prompt with benchmark info if provided
        system_prompt = self.config.system_prompt
        
        if benchmark_info:
            system_prompt += "\n\nAdditional context about the performance regression:\n"
            if "benchmark_name" in benchmark_info:
                system_prompt += f"- Benchmark: {benchmark_info['benchmark_name']}\n"
            if "parameters" in benchmark_info:
                params_str = ", ".join([f"{k}: {v}" for k, v in benchmark_info['parameters'].items()])
                system_prompt += f"- Parameters: {params_str}\n"
            if "metrics" in benchmark_info:
                metrics_str = ", ".join([f"{k}: {v}" for k, v in benchmark_info['metrics'].items()])
                system_prompt += f"- Performance metrics: {metrics_str}\n"
            if "regression_details" in benchmark_info:
                system_prompt += f"- Regression details: {benchmark_info['regression_details']}\n"
        
        try:
            # Call the AI model to analyze the diff
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": diff_content}
                ],
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            
            # Collect the response
            analysis_chunks = []
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    analysis_chunks.append(chunk.choices[0].delta.content)
                    
            return "".join(analysis_chunks)
                    
        except Exception as e:
            logger.error(f"Failed to analyze diff: {e}")
            return f"Error during analysis: {e}"
    
    async def analyze_commit(self, repo_path: str, commit_hash: str, benchmark_info: dict = None) -> DiffAnalysisResult:
        """Analyze a specific commit's diff.
        
        Args:
            repo_path: Path to the git repository
            commit_hash: Hash of the commit to analyze
            benchmark_info: Optional dictionary with benchmark information to provide context
            
        Returns:
            DiffAnalysisResult containing the commit hash, diff content, and analysis
        """
        # Get the diff for the commit
        diff_content = await self.get_commit_diff(repo_path, commit_hash)
        
        # Analyze the diff
        analysis = await self.analyze_diff(diff_content, benchmark_info)
        
        # Save to file if configured
        if self.config.output_file:
            output_path = self.config.output_file
            if "{commit_hash}" in str(output_path):
                output_path = Path(str(output_path).replace("{commit_hash}", commit_hash[:8]))
            
            # Create parent directories if they don't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write the analysis to the file
            with open(output_path, "w") as f:
                f.write(f"Commit: {commit_hash}\n\n")
                f.write("Diff:\n")
                f.write(diff_content)
                f.write("\n\nAnalysis:\n")
                f.write(analysis)
                
            logger.info(f"Analysis saved to {output_path}")
            
        return DiffAnalysisResult(
            commit_hash=commit_hash,
            diff_content=diff_content,
            analysis=analysis
        ) 