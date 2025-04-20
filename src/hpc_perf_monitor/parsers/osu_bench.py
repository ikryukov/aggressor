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
            if len(values) >= 2:  # At least size and latency
                # Verify values are numeric
                msg_size = float(values[0])
                latency = float(values[1])
                
                metrics = {
                    'msg_size': msg_size,
                    'latency': latency
                }
                
                # Add additional metrics if present (some OSU benchmarks provide more data)
                if len(values) >= 3:
                    metrics['bandwidth'] = float(values[2])
                
                logger.debug(f"Found valid data line: {metrics}")
                return True, metrics
            
            logger.debug(f"Line does not contain enough numeric values: {clean_line}")
            return False, None
            
        except (ValueError, IndexError) as e:
            logger.debug(f"Skipping line, not a data row: {e}")
            return False, None
    
    def parse(self, stdout: str, stderr: str) -> Dict[str, float]:
        """Parse benchmark output and extract metrics.
        
        Args:
            stdout: Standard output from benchmark
            stderr: Standard error from benchmark
            
        Returns:
            Dictionary containing extracted metrics
            
        The returned metrics include:
        - msg_size: Message size in bytes
        - latency: Latency in microseconds
        - bandwidth: Bandwidth in MB/s (if available)
        
        Raises:
            ValueError: If no valid data can be extracted from the output
        """
        logger.info("Starting to parse OSU benchmark output")
        if stderr:
            logger.warning("Stderr is not empty: %s", stderr)
            
        metrics: Dict[str, float] = {}
        found_data = False
        
        # Skip header lines and process data
        in_data_section = False
        lines_processed = 0
        
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
                # If this parser finds multiple data points, it takes the last one
                metrics = extracted_metrics
                found_data = True
        
        if not found_data:
            logger.warning(f"No valid data lines found in output after processing {lines_processed} lines")
            raise ValueError("Failed to extract metrics from OSU benchmark output")
            
        logger.info(
            f"Successfully parsed OSU benchmark results: "
            f"msg_size={metrics['msg_size']}, latency={metrics['latency']}"
        )
                
        return metrics 