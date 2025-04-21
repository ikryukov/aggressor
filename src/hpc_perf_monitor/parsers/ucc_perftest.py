"""Parser for UCC perftest benchmark results."""

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class UCCPerftestParser:
    """Parser for UCC perftest benchmark output.
    
    Parses the tabular output format that includes count, size, time, and bandwidth metrics.
    Each metric includes average, minimum, and maximum values.
    """
    
    @staticmethod
    def _extract_metrics_line(line: str) -> Tuple[bool, List[str]]:
        """Extract metrics from a line if it contains numeric data.
        
        Args:
            line: Line from benchmark output
            
        Returns:
            Tuple of (is_data_line, extracted_values)
        """
        logger.debug("Processing line: %s", line)
        
        # Remove [rank,x] prefix if present
        clean_line = re.sub(r'^\[\d+,\d+\]<stdout>:', '', line.strip())
        if clean_line != line.strip():
            logger.debug("Removed rank prefix. Clean line: %s", clean_line)
        
        # Split and filter out empty strings
        values = [v for v in clean_line.split() if v]
        logger.debug("Split values: %s", values)
        
        # Check if this is a data line (has enough numeric values)
        try:
            # Expect: count, size, avg_time, min_time, max_time, avg_bw, max_bw, min_bw
            if len(values) == 8:
                # Verify values are numeric
                [float(v) for v in values]
                logger.debug("Found valid data line with 8 numeric values")
                return True, values
            else:
                logger.debug("Line does not contain expected number of values (got %d, expected 8)", 
                           len(values))
        except ValueError as e:
            logger.debug("Line contains non-numeric values: %s", e)
        
        return False, []

    def parse(self, stdout: str, stderr: str) -> Dict[int, Dict[str, float]]:
        """Parse benchmark output and extract metrics.
        
        Args:
            stdout: Standard output from benchmark
            stderr: Standard error from benchmark
            
        Returns:
            Dictionary containing extracted metrics, organized by message size
            
        The returned structure is a dictionary where:
        - Key: Message size in bytes (as integer)
        - Value: Dictionary of metrics for that message size containing:
          - count: Number of elements
          - msg_size: Message size in bytes
          - latency_avg: Average latency in microseconds
          - latency_min: Minimum latency in microseconds
          - latency_max: Maximum latency in microseconds
          - bandwidth_avg: Average bandwidth in GB/s
          - bandwidth_min: Minimum bandwidth in GB/s
          - bandwidth_max: Maximum bandwidth in GB/s
        """
        logger.info("Starting to parse benchmark output")
        if stderr:
            logger.warning("Stderr is not empty: %s", stderr)
            
        results: Dict[int, Dict[str, float]] = {}
        lines_processed = 0
        
        for line in stdout.split('\n'):
            lines_processed += 1
            is_data, values = self._extract_metrics_line(line)
            if is_data:
                logger.debug("Found valid data line at line %d", lines_processed)
                msg_size = int(values[1])
                metrics = {
                    'count': int(values[0]),
                    'msg_size': msg_size,
                    'latency_avg': float(values[2]),
                    'latency_min': float(values[3]),
                    'latency_max': float(values[4]),
                    'bandwidth_avg': float(values[5]),
                    'bandwidth_max': float(values[6]),
                    'bandwidth_min': float(values[7])
                }
                logger.debug("Extracted metrics: %s", metrics)
                results[msg_size] = metrics
        
        if not results:
            logger.warning("No valid data lines found in output after processing %d lines", 
                         lines_processed)
        else:
            logger.info("Successfully parsed benchmark results with %d different message sizes", 
                      len(results))
                
        return results 