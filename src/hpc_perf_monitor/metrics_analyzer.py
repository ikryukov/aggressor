"""Performance metrics analysis functionality."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .benchmark_runner import BenchmarkResult
from .config import RegressionConfig

# Configure logging
logger = logging.getLogger(__name__)

class MetricsAnalyzer:
    """Analyzes benchmark results to detect performance regressions."""

    def __init__(self, config: RegressionConfig):
        """Initialize MetricsAnalyzer.

        Args:
            config: Regression detection configuration
        """
        self.config = config
        logger.info("Initialized MetricsAnalyzer with threshold: %f%%", self.config.threshold_pct)
        logger.debug("Metrics weights configuration: %s", self.config.metrics_weight)

    def _calculate_statistics(
        self,
        values: List[float]
    ) -> Tuple[float, float, float]:
        """Calculate basic statistics for a set of values.

        Args:
            values: List of metric values

        Returns:
            Tuple of (mean, std_dev, coefficient_of_variation)
        """
        if not values:
            logger.warning("Empty values list provided for statistics calculation")
            return 0.0, 0.0, 0.0

        values_array = np.array(values)
        mean = np.mean(values_array)
        std_dev = np.std(values_array)
        cv = (std_dev / mean) * 100 if mean != 0 else 0.0

        logger.debug("Statistics calculated - Mean: %f, StdDev: %f, CV: %f", mean, std_dev, cv)
        return mean, std_dev, cv

    def _detect_regression(
        self,
        ref_mean: float,
        test_mean: float,
        metric: str
    ) -> Tuple[bool, float]:
        """Detect if there is a regression between reference and test values.

        Args:
            ref_mean: Mean value from reference commit
            test_mean: Mean value from test commit
            metric: Name of the metric being compared

        Returns:
            Tuple of (is_regression, percentage_change)
        """
        if ref_mean == 0:
            logger.warning("Reference mean is 0 for metric '%s', skipping regression detection", metric)
            return False, 0.0

        pct_change = ((test_mean - ref_mean) / ref_mean) * 100
        weight = self.config.metrics_weight.get(metric, 1.0)
        
        logger.debug("Analyzing metric '%s' - Ref mean: %f, Test mean: %f, Change: %f%%, Weight: %f",
                    metric, ref_mean, test_mean, pct_change, weight)
        
        # For metrics where higher is better (e.g., bandwidth)
        if metric.lower() in ["bandwidth", "throughput", "ops_per_sec"]:
            is_regression = pct_change < -self.config.threshold_pct * weight
            if is_regression:
                logger.info("Performance regression detected for '%s' (higher is better): %f%% decrease",
                          metric, abs(pct_change))
        # For metrics where lower is better (e.g., latency)
        else:
            is_regression = pct_change > self.config.threshold_pct * weight
            if is_regression:
                logger.info("Performance regression detected for '%s' (lower is better): %f%% increase",
                          metric, pct_change)

        return is_regression, pct_change

    def analyze_results(
        self,
        ref_results: List[BenchmarkResult],
        test_results: List[BenchmarkResult]
    ) -> Dict:
        """Analyze benchmark results to detect regressions.

        Args:
            ref_results: List of BenchmarkResult objects from reference commit
            test_results: List of BenchmarkResult objects from test commit

        Returns:
            Dictionary containing analysis results with the following structure:
            {
                "regressions": [
                    {
                        "benchmark": str,
                        "parameters": Dict[str, Any],
                        "metric": str,
                        "reference": {"mean": float, "std_dev": float, "cv": float},
                        "test": {"mean": float, "std_dev": float, "cv": float},
                        "percent_change": float
                    },
                    ...
                ],
                "all_results": [
                    {
                        "benchmark": str,
                        "parameters": Dict[str, Any],
                        "metric": str,
                        "reference": {"mean": float, "std_dev": float, "cv": float},
                        "test": {"mean": float, "std_dev": float, "cv": float},
                        "percent_change": float,
                        "is_regression": bool,
                        "latency_pct_change": float,   # Percentage change for latency
                        "bandwidth_pct_change": float  # Percentage change for bandwidth
                    },
                    ...
                ],
                "summary": {
                    "total_comparisons": int,
                    "regressions_found": int,
                    "ref_commit": str,
                    "test_commit": str
                }
            }

        Raises:
            ValueError: If insufficient runs or invalid data is provided
        """
        # Input validation
        if not ref_results or not test_results:
            raise ValueError("Empty results provided for analysis")

        logger.info("Starting benchmark analysis - Reference commit: %s, Test commit: %s",
                   ref_results[0].commit_hash, test_results[0].commit_hash)

        if len(ref_results) < self.config.min_runs or len(test_results) < self.config.min_runs:
            error_msg = f"Insufficient runs for analysis. Need at least {self.config.min_runs} runs."
            logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            # Convert BenchmarkResults to a more analyzable format
            ref_data = []
            test_data = []
            
            for result in ref_results:
                row = {
                    "benchmark_name": result.benchmark_name,
                    **result.parameters,
                    **{f"metrics.{k}": v for k, v in result.metrics.items()}
                }
                ref_data.append(row)
            
            for result in test_results:
                row = {
                    "benchmark_name": result.benchmark_name,
                    **result.parameters,
                    **{f"metrics.{k}": v for k, v in result.metrics.items()}
                }
                test_data.append(row)

            # Convert to DataFrames
            ref_df = pd.DataFrame(ref_data)
            test_df = pd.DataFrame(test_data)

            # Get group columns (benchmark name and parameters)
            param_cols = list(ref_results[0].parameters.keys())
            group_cols = ["benchmark_name"] + param_cols
            logger.debug("Grouping results by columns: %s", group_cols)
            
            analysis = {
                "regressions": [],
                "all_results": [],
                "summary": {
                    "total_comparisons": 0,
                    "regressions_found": 0,
                    "ref_commit": ref_results[0].commit_hash,
                    "test_commit": test_results[0].commit_hash
                }
            }

            # Analyze each group
            for name, ref_group in ref_df.groupby(group_cols):
                logger.debug("Analyzing benchmark group: %s", name)
                
                # Find matching test group
                test_group = test_df[
                    test_df[group_cols].apply(tuple, axis=1) == tuple(name)
                ]
                
                if test_group.empty:
                    logger.warning("No matching test data found for benchmark group: %s", name)
                    continue

                # Get metrics from first result
                metrics = ref_results[0].metrics.keys()
                if not metrics:
                    logger.warning("No metrics found for benchmark group: %s", name)
                    continue

                # Analyze each metric
                for metric in metrics:
                    try:
                        logger.debug("Analyzing metric '%s' for benchmark group: %s", metric, name)
                        
                        # Extract metric values safely
                        metric_col = f"metrics.{metric}"
                        ref_values = ref_group[metric_col].dropna().tolist()
                        test_values = test_group[metric_col].dropna().tolist()

                        if not ref_values or not test_values:
                            logger.warning("Empty metric values for '%s' in group: %s", metric, name)
                            continue

                        ref_mean, ref_std, ref_cv = self._calculate_statistics(ref_values)
                        test_mean, test_std, test_cv = self._calculate_statistics(test_values)

                        is_regression, pct_change = self._detect_regression(
                            ref_mean, test_mean, metric
                        )

                        analysis["summary"]["total_comparisons"] += 1

                        # Convert parameters to a dictionary, handling enum values
                        params = {}
                        for i, col in enumerate(param_cols):
                            val = name[i + 1]  # +1 because name[0] is benchmark_name
                            # Handle enum values by getting their value
                            params[col] = getattr(val, 'value', val)

                        # Calculate percentage differences for both latency and bandwidth
                        latency_pct_change = ((test_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0.0
                        bandwidth_pct_change = ((test_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0.0

                        # Create result info for both regressions and all results
                        result_info = {
                            "benchmark": name[0],
                            "parameters": params,
                            "metric": metric,
                            "reference": {
                                "mean": float(ref_mean),
                                "std_dev": float(ref_std),
                                "cv": float(ref_cv)
                            },
                            "test": {
                                "mean": float(test_mean),
                                "std_dev": float(test_std),
                                "cv": float(test_cv)
                            },
                            "percent_change": float(pct_change),
                            "latency_pct_change": float(latency_pct_change),
                            "bandwidth_pct_change": float(bandwidth_pct_change),
                            "is_regression": is_regression
                        }
                        
                        # Add to all_results
                        analysis["all_results"].append(result_info)
                        
                        # If it's a regression, also add to the regressions list
                        if is_regression:
                            analysis["summary"]["regressions_found"] += 1
                            # Copy without the is_regression flag for backward compatibility
                            regression_info = {k: v for k, v in result_info.items() if k != "is_regression"}
                            analysis["regressions"].append(regression_info)
                            logger.warning("Regression details: %s", regression_info)
                    
                    except Exception as e:
                        logger.error("Error analyzing metric '%s' for group '%s': %s", 
                                   metric, name, str(e))
                        continue

            logger.info("Analysis complete - Total comparisons: %d, Regressions found: %d",
                       analysis["summary"]["total_comparisons"],
                       analysis["summary"]["regressions_found"])
            return analysis

        except Exception as e:
            error_msg = f"Failed to analyze results: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg) from e

    def save_analysis(self, analysis: Dict, output_file: Path) -> None:
        """Save analysis results to a file.

        Args:
            analysis: Analysis results dictionary
            output_file: Path to save results to
        """
        logger.info("Saving analysis results to: %s", output_file)
        try:
            with output_file.open("w") as f:
                json.dump(analysis, f, indent=2)
            logger.debug("Analysis results successfully saved")
        except Exception as e:
            logger.error("Failed to save analysis results: %s", str(e))
            raise

    def load_analysis(self, input_file: Path) -> Dict:
        """Load analysis results from a file.

        Args:
            input_file: Path to load results from

        Returns:
            Analysis results dictionary
        """
        logger.info("Loading analysis results from: %s", input_file)
        try:
            with input_file.open("r") as f:
                analysis = json.load(f)
            logger.debug("Analysis results successfully loaded")
            return analysis
        except Exception as e:
            logger.error("Failed to load analysis results: %s", str(e))
            raise 