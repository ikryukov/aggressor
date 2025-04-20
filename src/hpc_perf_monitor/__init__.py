"""HPC Performance Monitor - A tool for detecting performance regressions in HPC middleware."""

import logging
import sys

# Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler('hpc_perf_monitor.log')
#     ]
# )

# Optionally set more verbose logging for the metrics analyzer
# logger = logging.getLogger("hpc_perf_monitor.metrics_analyzer")
# logger.setLevel(logging.DEBUG)

__version__ = "0.1.0" 