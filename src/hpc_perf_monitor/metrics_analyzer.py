"""Performance metrics analysis functionality."""

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .benchmark_runner import BenchmarkResult
from .config import RegressionConfig

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of benchmark analysis."""
    name: str
    parameters: dict[str, Any]
    data: pd.DataFrame
    significant_changes: dict[str, list[dict[str, Any]]]  # Dictionary of metric name to list of significant changes
    has_regression: bool  # Flag indicating if any regression was found

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

    def _process_result(self, ref_result: BenchmarkResult, test_result: BenchmarkResult) -> AnalysisResult:
        """Process benchmark results and calculate performance differences.

        Args:
            ref_result: Reference benchmark result
            test_result: Test benchmark result

        Returns:
            AnalysisResult containing benchmark name, parameters and comparison data
        """
        comparison_data = []
        significant_changes = {}
        has_regression = False
        
        # Process each message size
        for msg_size in ref_result.metrics.keys():
            ref_metrics = ref_result.metrics[msg_size]
            test_metrics = test_result.metrics[msg_size]
            
            # Calculate percentage differences for latency
            latency_avg_diff = -((test_metrics['latency_avg'] - ref_metrics['latency_avg']) / 
                          ref_metrics['latency_avg']) * 100
            latency_min_diff = -((test_metrics['latency_min'] - ref_metrics['latency_min']) / 
                          ref_metrics['latency_min']) * 100
            latency_max_diff = -((test_metrics['latency_max'] - ref_metrics['latency_max']) / 
                          ref_metrics['latency_max']) * 100

            # Initialize comparison entry with common metrics
            comparison_entry = {
                'count': ref_metrics['count'],
                'msg_size': msg_size,
                'ref_latency_avg': ref_metrics['latency_avg'],
                'test_latency_avg': test_metrics['latency_avg'],
                'ref_latency_min': ref_metrics['latency_min'],
                'test_latency_min': test_metrics['latency_min'],
                'ref_latency_max': ref_metrics['latency_max'],
                'test_latency_max': test_metrics['latency_max'],
                'latency_avg_diff_pct': latency_avg_diff,
                'latency_min_diff_pct': latency_min_diff,
                'latency_max_diff_pct': latency_max_diff,
            }
            
            # Add bandwidth metrics if they exist
            if 'bandwidth_avg' in ref_metrics and 'bandwidth_avg' in test_metrics:
                bandwidth_diff = ((test_metrics['bandwidth_avg'] - ref_metrics['bandwidth_avg']) / 
                                ref_metrics['bandwidth_avg']) * 100 if ref_metrics['bandwidth_avg'] != 0 else 0
                comparison_entry.update({
                    'ref_bandwidth_avg': ref_metrics['bandwidth_avg'],
                    'test_bandwidth_avg': test_metrics['bandwidth_avg'],
                    'bandwidth_diff_pct': bandwidth_diff,
                })
            
            comparison_data.append(comparison_entry)
        
        # Create DataFrame and sort by message size
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('msg_size')
        
        df['count'] = df['count'].astype(int)
        df['msg_size'] = df['msg_size'].astype(int)

        # Track significant changes and regression status
        for watch_metric in ref_result.metrics_watch:
            diff_name = f"{watch_metric}_diff_pct"
            if diff_name in df.columns:
                # Find significant changes
                significant_rows = df[abs(df[diff_name]) > self.config.threshold_pct]
                if not significant_rows.empty:
                    significant_changes[watch_metric] = significant_rows.to_dict(orient='records')
                    # Check for regression (negative difference)
                    if df[diff_name].min() < -self.config.threshold_pct:
                        has_regression = True
                        logger.warning(f"Metric {watch_metric} has a difference of {df[diff_name].min()}%")
        
        return AnalysisResult(
            name=ref_result.benchmark_name,
            parameters=ref_result.parameters,
            data=df,
            significant_changes=significant_changes,
            has_regression=has_regression
        )

    def analyze_results(
        self,
        ref_results: list[BenchmarkResult],
        test_results: list[BenchmarkResult]
    ) -> list[AnalysisResult]:
        """Analyze benchmark results and calculate performance differences.

        Args:
            ref_results: List of reference benchmark results
            test_results: List of test benchmark results to compare against reference

        Returns:
            List of AnalysisResult containing comparison results with percentage differences
        """
        analysis_results = []
        for ref_result, test_result in zip(ref_results, test_results, strict=False):
            analysis_results.append(self._process_result(ref_result, test_result))

        return analysis_results
