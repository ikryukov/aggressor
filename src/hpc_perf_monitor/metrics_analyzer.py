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

    def analyze_results(
        self,
        ref_results: List[BenchmarkResult],
        test_results: List[BenchmarkResult]
    ) -> pd.DataFrame:
        """Analyze benchmark results and calculate performance differences.

        Args:
            ref_results: List of reference benchmark results
            test_results: List of test benchmark results to compare against reference

        Returns:
            DataFrame containing comparison results with percentage differences
        """
        comparison_data = []
        
        # Process each message size
        for msg_size in ref_results[0].metrics.keys():
            ref_metrics = ref_results[0].metrics[msg_size]
            test_metrics = test_results[0].metrics[msg_size]
            
            # Calculate percentage differences
            latency_diff = ((test_metrics['latency_avg'] - ref_metrics['latency_avg']) / 
                          ref_metrics['latency_avg']) * 100
            bandwidth_diff = ((test_metrics['bandwidth_avg'] - ref_metrics['bandwidth_avg']) / 
                            ref_metrics['bandwidth_avg']) * 100 if ref_metrics['bandwidth_avg'] != 0 else 0
            
            comparison_data.append({
                'count': ref_metrics['count'],
                'msg_size': msg_size,
                'ref_latency_avg': ref_metrics['latency_avg'],
                'test_latency_avg': test_metrics['latency_avg'],
                'latency_diff_pct': latency_diff,
                'ref_bandwidth_avg': ref_metrics['bandwidth_avg'],
                'test_bandwidth_avg': test_metrics['bandwidth_avg'],
                'bandwidth_diff_pct': bandwidth_diff,
                
            })
        
        # Create DataFrame and sort by message size
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('msg_size')
        
        df['count'] = df['count'].astype(int)
        df['msg_size'] = df['msg_size'].astype(int)

        # Log significant differences
        significant_latency = df[abs(df['latency_diff_pct']) > self.config.threshold_pct]
        if not significant_latency.empty:
            logger.warning("Significant latency differences found:\n%s", 
                         significant_latency[['msg_size', 'latency_diff_pct']])
        
        return df

