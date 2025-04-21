"""Benchmark output parsers."""

import logging
from typing import Dict, Protocol, Type, Union

from .osu_bench import OSUBenchParser
from .ucc_perftest import UCCPerftestParser

logger = logging.getLogger(__name__)


class BenchmarkParser(Protocol):
    """Protocol defining the interface for benchmark parsers."""
    
    def parse(self, stdout: str, stderr: str) -> dict[str, float]:
        """Parse benchmark output and extract metrics.
        
        Args:
            stdout: Standard output from benchmark
            stderr: Standard error from benchmark
            
        Returns:
            Dictionary containing extracted metrics
        """
        ...


# Registry of available parsers
PARSERS: dict[str, type[BenchmarkParser]] = {
    'ucc_perftest': UCCPerftestParser,
    'osu_bench': OSUBenchParser
}


def get_parser(parser_name: str) -> BenchmarkParser:
    """Get parser instance by name.
    
    Args:
        parser_name: Name of the parser to get
        
    Returns:
        Parser instance
        
    Raises:
        KeyError: If parser_name is not found in registry
    """
    logger.debug("Requested parser: %s", parser_name)
    logger.debug("Available parsers: %s", list(PARSERS.keys()))
    
    if parser_name not in PARSERS:
        logger.error("Parser '%s' not found in registry", parser_name)
        raise KeyError(f"Parser '{parser_name}' not found. Available parsers: {list(PARSERS.keys())}")
    
    logger.info("Creating parser instance for '%s'", parser_name)
    return PARSERS[parser_name]() 