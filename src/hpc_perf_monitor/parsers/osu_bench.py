"""Parser for OSU benchmark results."""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OSUBenchParser:
    """Parser for OSU MPI benchmark outputs.
    
    Parses the tabular output format that typically includes size and latency metrics.
    This parser is designed to handle output from the OSU MPI Micro-benchmarks suite,
    which typically presents data in a tabular format with headers followed by rows
    of size and latency values.
    """
    
    def _extract_data_line(self, line: str) -> Tuple[bool, Optional[Dict[str, float]]]:
        """Extract metrics from a line if it contains valid OSU benchmark data.
        
        Args:
            line: A line from the benchmark output
            
        Returns:
            Tuple of (is_data_line, extracted_metrics_dict)
        """
        # Clean the line (remove MPI rank prefixes if present)
        clean_line = re.sub(r'^\[\d+,\d+\]<stdout>:', '', line.strip())
        
        # Skip empty lines
        if not clean_line:
            return False, None
            
        # Try to parse a data line
        try:
            values = [v for v in clean_line.split() if v]
            if len(values) >= 5:  # Size, Avg Latency, Min Latency, Max Latency, Iterations
                # Verify values are numeric
                msg_size = int(values[0])
                avg_latency = float(values[1])
                min_latency = float(values[2])
                max_latency = float(values[3])
                iterations = int(values[4])
                
                metrics = {
                    'msg_size': msg_size,
                    'latency_avg': avg_latency,
                    'latency_min': min_latency,
                    'latency_max': max_latency,
                    'count': iterations
                }
                
                logger.debug(f"Found valid data line: {metrics}")
                return True, metrics
            
            logger.debug(f"Line does not contain enough numeric values: {clean_line}")
            return False, None
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping line, not a data row: {e}")
            return False, None
    
    def parse(self, stdout: str, stderr: str) -> Dict[int, Dict[str, float]]:
        """Parse benchmark output and extract metrics.
        
        Args:
            stdout: Standard output from benchmark
            stderr: Standard error from benchmark
            
        Returns:
            Dictionary containing extracted metrics, organized by message size
            
        The returned metrics for each message size include:
        - msg_size: Message size in bytes
        - latency_avg: Average latency in microseconds
        - latency_min: Minimum latency in microseconds
        - latency_max: Maximum latency in microseconds
        - count: Number of iterations performed
        
        Raises:
            ValueError: If no valid data can be extracted from the output
        """
        logger.info("Starting to parse OSU benchmark output")
        if stderr:
            logger.warning("Stderr is not empty: %s", stderr)
            
        metrics_by_size: Dict[int, Dict[str, float]] = {}
        found_data = False
        lines_processed = 0
        
        # Skip header lines and process data
        in_data_section = False
        
        for line in stdout.split('\n'):
            lines_processed += 1
            
            # Skip header sections and find the start of the data
            if "Size" in line and "Latency" in line:
                in_data_section = True
                logger.debug(f"Found header line at line {lines_processed}: {line}")
                continue
                
            if not in_data_section:
                continue
            
            # Try to parse data
            is_data, extracted_metrics = self._extract_data_line(line)
            if is_data and extracted_metrics is not None:
                msg_size = int(extracted_metrics['msg_size'])
                metrics_by_size[msg_size] = extracted_metrics
                found_data = True
                logger.debug("Found valid data line at line %d: %s", lines_processed, extracted_metrics)
        
        if not found_data:
            logger.warning("No valid data lines found in output after processing %d lines", lines_processed)
            raise ValueError("Failed to extract metrics from OSU benchmark output")
            
        logger.info("Successfully parsed OSU benchmark results with %d different message sizes", 
                  len(metrics_by_size))
                
        return metrics_by_size 