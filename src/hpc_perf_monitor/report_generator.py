"""Report generation functionality."""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from jinja2 import Environment, FileSystemLoader


class ReportGenerator:
    """Generates formatted reports from analysis results."""

    def __init__(self):
        """Initialize ReportGenerator."""
        self.env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))

    def _format_metric_value(self, value: float, metric: str) -> str:
        """Format metric value with appropriate units.

        Args:
            value: Metric value to format
            metric: Name of the metric

        Returns:
            Formatted string with units
        """
        if metric.lower() in ["bandwidth", "throughput"]:
            if value > 1e9:
                return f"{value/1e9:.2f} GB/s"
            elif value > 1e6:
                return f"{value/1e6:.2f} MB/s"
            else:
                return f"{value/1e3:.2f} KB/s"
        elif metric.lower() == "latency":
            if value < 1e-6:
                return f"{value*1e9:.2f} ns"
            elif value < 1e-3:
                return f"{value*1e6:.2f} µs"
            else:
                return f"{value*1e3:.2f} ms"
        else:
            return f"{value:.2f}"

    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert NumPy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(item) for item in obj)
        elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return obj

    def _is_improvement(self, percent_change: float, metric: str) -> bool:
        """Determine if a change represents an improvement.

        Args:
            percent_change: The percentage change value
            metric: Name of the metric

        Returns:
            True if the change is an improvement, False otherwise
        """
        # For metrics where higher is better (e.g., bandwidth)
        if metric.lower() in ["bandwidth", "throughput", "ops_per_sec"]:
            return percent_change > 0
        # For metrics where lower is better (e.g., latency)
        else:
            return percent_change < 0
    
    def _calculate_min_max_values(self, mean: float, std_dev: float) -> Tuple[float, float]:
        """Calculate min and max values based on mean and standard deviation.
        
        Args:
            mean: The mean value
            std_dev: The standard deviation
            
        Returns:
            Tuple of (min_value, max_value)
        """
        return (mean - std_dev, mean + std_dev)

    def _prepare_template_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare and enhance the analysis data for templates.

        Args:
            analysis: Raw analysis results dictionary

        Returns:
            Enhanced analysis dictionary with additional formatting data
        """
        enhanced_analysis = analysis.copy()
        
        # Regroup results by benchmark name and parameters (except msg_size)
        if "all_results" in enhanced_analysis:
            # Create a dictionary to hold grouped results
            grouped_results = {}
            
            # First pass: Group results by benchmark name and parameters (except msg_size)
            for result in enhanced_analysis["all_results"]:
                # Create a key that excludes msg_size
                param_key_items = []
                msg_size = None
                for param, value in result["parameters"].items():
                    if param == "msg_size":
                        # Convert NumPy types to native Python types
                        if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
                            msg_size = int(value)
                        elif isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
                            msg_size = float(value)
                        else:
                            msg_size = value
                    else:
                        param_key_items.append(f"{param}={value}")
                
                # Skip results without msg_size parameter
                if msg_size is None:
                    continue
                
                # Generate the unique key for this benchmark + parameters combination
                benchmark_key = f"{result['benchmark']}|{'|'.join(sorted(param_key_items))}"
                
                # Initialize the group if it doesn't exist
                if benchmark_key not in grouped_results:
                    grouped_results[benchmark_key] = {
                        "benchmark": result["benchmark"],
                        "parameters": {k: v for k, v in result["parameters"].items() if k != "msg_size"},
                        "metrics": {},
                        "has_regression": False
                    }
                
                # Initialize the metric if it doesn't exist
                metric = result["metric"]
                if metric not in grouped_results[benchmark_key]["metrics"]:
                    grouped_results[benchmark_key]["metrics"][metric] = {}
                
                # Add the result data for this msg_size
                grouped_results[benchmark_key]["metrics"][metric][msg_size] = {
                    "reference": result["reference"],
                    "test": result["test"],
                    "percent_change": result["percent_change"],
                    "latency_pct_change": result.get("latency_pct_change", 0.0),
                    "bandwidth_pct_change": result.get("bandwidth_pct_change", 0.0),
                    "is_regression": result["is_regression"]
                }
                
                # Update the regression flag
                if result["is_regression"]:
                    grouped_results[benchmark_key]["has_regression"] = True
            
            # Convert grouped_results to a list for the template
            enhanced_analysis["grouped_experiments"] = list(grouped_results.values())
        
        # Process all_results if present (original formatting code)
        if "all_results" in enhanced_analysis:
            for result in enhanced_analysis["all_results"]:
                # Add formatted std_dev values if not present
                if "std_dev" in result["reference"] and "formatted_std_dev" not in result["reference"]:
                    result["reference"]["formatted_std_dev"] = self._format_metric_value(
                        result["reference"]["std_dev"],
                        result["metric"]
                    )
                
                if "std_dev" in result["test"] and "formatted_std_dev" not in result["test"]:
                    result["test"]["formatted_std_dev"] = self._format_metric_value(
                        result["test"]["std_dev"],
                        result["metric"]
                    )
                
                # Add formatted mean values if not present
                if "mean" in result["reference"] and "formatted_mean" not in result["reference"]:
                    result["reference"]["formatted_mean"] = self._format_metric_value(
                        result["reference"]["mean"],
                        result["metric"]
                    )
                
                if "mean" in result["test"] and "formatted_mean" not in result["test"]:
                    result["test"]["formatted_mean"] = self._format_metric_value(
                        result["test"]["mean"],
                        result["metric"]
                    )
                
                # Add min/max values for reference and test
                if "mean" in result["reference"] and "std_dev" in result["reference"]:
                    min_val, max_val = self._calculate_min_max_values(
                        result["reference"]["mean"],
                        result["reference"]["std_dev"]
                    )
                    result["reference"]["min"] = min_val
                    result["reference"]["max"] = max_val
                    result["reference"]["formatted_min"] = self._format_metric_value(min_val, result["metric"])
                    result["reference"]["formatted_max"] = self._format_metric_value(max_val, result["metric"])
                
                if "mean" in result["test"] and "std_dev" in result["test"]:
                    min_val, max_val = self._calculate_min_max_values(
                        result["test"]["mean"],
                        result["test"]["std_dev"]
                    )
                    result["test"]["min"] = min_val
                    result["test"]["max"] = max_val
                    result["test"]["formatted_min"] = self._format_metric_value(min_val, result["metric"])
                    result["test"]["formatted_max"] = self._format_metric_value(max_val, result["metric"])
        
        # Process regressions for backward compatibility
        for regression in enhanced_analysis.get("regressions", []):
            # Add formatted std_dev values if not present
            if "std_dev" in regression["reference"] and "formatted_std_dev" not in regression["reference"]:
                regression["reference"]["formatted_std_dev"] = self._format_metric_value(
                    regression["reference"]["std_dev"],
                    regression["metric"]
                )
            
            if "std_dev" in regression["test"] and "formatted_std_dev" not in regression["test"]:
                regression["test"]["formatted_std_dev"] = self._format_metric_value(
                    regression["test"]["std_dev"],
                    regression["metric"]
                )
            
            # Add formatted mean values if not present
            if "mean" in regression["reference"] and "formatted_mean" not in regression["reference"]:
                regression["reference"]["formatted_mean"] = self._format_metric_value(
                    regression["reference"]["mean"],
                    regression["metric"]
                )
            
            if "mean" in regression["test"] and "formatted_mean" not in regression["test"]:
                regression["test"]["formatted_mean"] = self._format_metric_value(
                    regression["test"]["mean"],
                    regression["metric"]
                )
            
            # Add min/max values for reference and test
            if "mean" in regression["reference"] and "std_dev" in regression["reference"]:
                min_val, max_val = self._calculate_min_max_values(
                    regression["reference"]["mean"],
                    regression["reference"]["std_dev"]
                )
                regression["reference"]["min"] = min_val
                regression["reference"]["max"] = max_val
                regression["reference"]["formatted_min"] = self._format_metric_value(min_val, regression["metric"])
                regression["reference"]["formatted_max"] = self._format_metric_value(max_val, regression["metric"])
            
            if "mean" in regression["test"] and "std_dev" in regression["test"]:
                min_val, max_val = self._calculate_min_max_values(
                    regression["test"]["mean"],
                    regression["test"]["std_dev"]
                )
                regression["test"]["min"] = min_val
                regression["test"]["max"] = max_val
                regression["test"]["formatted_min"] = self._format_metric_value(min_val, regression["metric"])
                regression["test"]["formatted_max"] = self._format_metric_value(max_val, regression["metric"])
            
            # Add improvement flag
            regression["is_improvement"] = self._is_improvement(
                regression["percent_change"],
                regression["metric"]
            )
        
        return enhanced_analysis

    def generate_markdown(
        self, 
        analysis: Dict, 
        output_file: Path, 
        commit_info: Optional[Dict] = None
    ) -> None:
        """Generate markdown report.

        Args:
            analysis: Analysis results dictionary
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        template = self.env.get_template("report.md.j2")
        enhanced_analysis = self._prepare_template_data(analysis)
        
        report = template.render(
            analysis=enhanced_analysis,
            format_metric=self._format_metric_value,
            is_improvement=self._is_improvement,
            commit_info=commit_info
        )
        output_file.write_text(report)

    def generate_html(
        self, 
        analysis: Dict, 
        output_file: Path, 
        commit_info: Optional[Dict] = None
    ) -> None:
        """Generate HTML report.

        Args:
            analysis: Analysis results dictionary
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        template = self.env.get_template("report.html.j2")
        enhanced_analysis = self._prepare_template_data(analysis)
        
        report = template.render(
            analysis=enhanced_analysis,
            format_metric=self._format_metric_value,
            is_improvement=self._is_improvement,
            commit_info=commit_info
        )
        output_file.write_text(report)

    def generate_json(
        self, 
        analysis: Dict, 
        output_file: Path, 
        commit_info: Optional[Dict] = None
    ) -> None:
        """Generate JSON report.

        Args:
            analysis: Analysis results dictionary
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        # Prepare enhanced analysis
        enhanced_analysis = self._prepare_template_data(analysis)
        
        # Add commit info if provided
        if commit_info:
            enhanced_analysis["commit_info"] = commit_info

        # Convert NumPy types to Python native types for JSON serialization
        serializable_analysis = self._convert_to_serializable(enhanced_analysis)
        
        with output_file.open("w") as f:
            json.dump(serializable_analysis, f, indent=2)

    def generate_report(
        self,
        analysis: Dict,
        output_file: Path,
        format: str = "markdown",
        commit_info: Optional[Dict] = None
    ) -> None:
        """Generate report in specified format.

        Args:
            analysis: Analysis results dictionary
            output_file: Path to save report to
            format: Report format (markdown, html, or json)
            commit_info: Optional dictionary with commit information

        Raises:
            ValueError: If format is not supported
        """
        if format == "markdown":
            self.generate_markdown(analysis, output_file, commit_info)
        elif format == "html":
            self.generate_html(analysis, output_file, commit_info)
        elif format == "json":
            self.generate_json(analysis, output_file, commit_info)
        else:
            raise ValueError(f"Unsupported report format: {format}") 