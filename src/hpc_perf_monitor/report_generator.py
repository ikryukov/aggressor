"""Report generation functionality."""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

from jinja2 import Environment, FileSystemLoader

class ReportGenerator:
    """Generates formatted reports from analysis results."""

    def __init__(self):
        """Initialize ReportGenerator."""
        self.env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))

    def _style_analysis_table(self, df: pd.DataFrame) -> str:
        """Return HTML table with negative numbers styled in red."""
        
        column_names = {
            'count': 'Count',
            'msg_size': 'Message Size',
            'ref_latency_avg': 'Reference Latency (avg)',
            'test_latency_avg': 'Test Latency (avg)',
            'latency_diff_pct': 'Latency Difference (%)',
            'ref_bandwidth_avg': 'Reference Bandwidth GB/s (avg)',
            'test_bandwidth_avg': 'Test Bandwidth GB/s (avg)',
            'bandwidth_diff_pct': 'Bandwidth Difference (%)'
        }
        
        df_display = df.rename(columns=column_names)
        df_display = df_display.reset_index(drop=True)

        def color_negative(val):
            if isinstance(val, (int, float)) and val < 0:
                return "color: red"
            return ""

        df_display = df_display.round(2)

        styled = (
            df_display.style
            .map(color_negative)
            .set_table_attributes('class="table table-striped"')
        )

        return styled.to_html(index=False, float_format="{0:.2f}".format)

    def generate_report(
        self,
        analysis: pd.DataFrame,
        output_file: Path,
        format: str = "markdown",
        commit_info: Optional[Dict] = None
    ) -> None:
        """Generate report in specified format.

        Args:
            analysis: Analysis results DataFrame
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

    def generate_markdown(self, analysis: pd.DataFrame, output_file: Path, commit_info: Optional[Dict] = None) -> None:
        """Generate markdown report.

        Args:
            analysis: Analysis results DataFrame
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        template = self.env.get_template("report.md.j2")
        
        # Prepare data for template
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "commit_info": commit_info or {},
            "summary": self._generate_summary(analysis),
            "table": analysis.to_markdown(index=False),
            "significant_changes": self._get_significant_changes(analysis)
        }
        
        with open(output_file, "w") as f:
            f.write(template.render(**report_data))

    def generate_html(self, analysis: pd.DataFrame, output_file: Path, commit_info: Optional[Dict] = None) -> None:
        """Generate HTML report.

        Args:
            analysis: Analysis results DataFrame
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        template = self.env.get_template("report.html.j2")
        
        # Prepare data for template
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "commit_info": commit_info or {},
            "summary": self._generate_summary(analysis),
            "table": self._style_analysis_table(analysis),
            "significant_changes": self._get_significant_changes(analysis),
            "charts": {
                "latency": self._generate_latency_chart_data(analysis),
                "bandwidth": self._generate_bandwidth_chart_data(analysis)
            }
        }
        
        with open(output_file, "w") as f:
            f.write(template.render(**report_data))

    def generate_json(self, analysis: pd.DataFrame, output_file: Path, commit_info: Optional[Dict] = None) -> None:
        """Generate JSON report.

        Args:
            analysis: Analysis results DataFrame
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "commit_info": commit_info or {},
            "summary": self._generate_summary(analysis),
            "data": analysis.to_dict(orient="records"),
            "significant_changes": self._get_significant_changes(analysis)
        }
        
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

    def _generate_summary(self, analysis: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics from analysis data.

        Args:
            analysis: Analysis results DataFrame

        Returns:
            Dictionary containing summary statistics
        """
        return {
            "total_tests": len(analysis),
            "avg_latency_diff": analysis["latency_diff_pct"].mean(),
            "max_latency_diff": analysis["latency_diff_pct"].max(),
            "min_latency_diff": analysis["latency_diff_pct"].min(),
            "avg_bandwidth_diff": analysis["bandwidth_diff_pct"].mean(),
            "max_bandwidth_diff": analysis["bandwidth_diff_pct"].max(),
            "min_bandwidth_diff": analysis["bandwidth_diff_pct"].min()
        }

    def _get_significant_changes(self, analysis: pd.DataFrame, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Get significant changes exceeding threshold.

        Args:
            analysis: Analysis results DataFrame
            threshold: Threshold for significant changes (percentage)

        Returns:
            List of dictionaries containing significant changes
        """
        significant = analysis[
            (abs(analysis["latency_diff_pct"]) > threshold) |
            (abs(analysis["bandwidth_diff_pct"]) > threshold)
        ]
        
        return significant.to_dict(orient="records")

    def _generate_latency_chart_data(self, analysis: pd.DataFrame) -> Dict[str, List[float]]:
        """Generate data for latency chart.

        Args:
            analysis: Analysis results DataFrame

        Returns:
            Dictionary with chart data
        """
        return {
            "msg_sizes": analysis["msg_size"].tolist(),
            "ref_latency": analysis["ref_latency_avg"].tolist(),
            "test_latency": analysis["test_latency_avg"].tolist(),
            "diff_pct": analysis["latency_diff_pct"].tolist()
        }

    def _generate_bandwidth_chart_data(self, analysis: pd.DataFrame) -> Dict[str, List[float]]:
        """Generate data for bandwidth chart.

        Args:
            analysis: Analysis results DataFrame

        Returns:
            Dictionary with chart data
        """
        return {
            "msg_sizes": analysis["msg_size"].tolist(),
            "ref_bandwidth": analysis["ref_bandwidth_avg"].tolist(),
            "test_bandwidth": analysis["test_bandwidth_avg"].tolist(),
            "diff_pct": analysis["bandwidth_diff_pct"].tolist()
        } 