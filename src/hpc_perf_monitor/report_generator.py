"""Report generation functionality."""

import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

from jinja2 import Environment, FileSystemLoader
from .metrics_analyzer import AnalysisResult

logger = logging.getLogger(__name__)

class ReportGenerator:
    """Generates formatted reports from analysis results."""

    def __init__(self):
        """Initialize ReportGenerator."""
        self.env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"))

    def _style_analysis_table(self, df: pd.DataFrame) -> str:
        """Return HTML table with negative numbers styled in red."""
        
        # Create a copy of the DataFrame to avoid modifying the original
        df_display = df.copy()
        
        # Rename columns for display
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
        
        # Only rename columns that exist in the DataFrame
        existing_columns = {k: v for k, v in column_names.items() if k in df_display.columns}
        df_display = df_display.rename(columns=existing_columns)
        
        # Reset index and round numeric columns
        df_display = df_display.reset_index(drop=True)
        numeric_columns = df_display.select_dtypes(include=['float64', 'int64']).columns
        df_display[numeric_columns] = df_display[numeric_columns].round(2)

        def color_negative(val):
            if isinstance(val, (int, float)) and val < 0:
                return "color: red"
            return ""

        styled = (
            df_display.style
            .map(color_negative)
            .set_table_attributes('class="table table-striped"')
        )

        return styled.to_html(index=False, float_format="{0:.2f}".format)

    def generate_report(
        self,
        analysis: List[AnalysisResult],
        output_file: Path,
        format: str = "markdown",
        commit_info: Optional[Dict] = None
    ) -> None:
        """Generate report in specified format.

        Args:
            analysis: List of analysis results
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

    def generate_markdown(self, analysis: List[AnalysisResult], output_file: Path, commit_info: Optional[Dict] = None) -> None:
        """Generate markdown report.

        Args:
            analysis: List of analysis results
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        template = self.env.get_template("report.md.j2")
        
        # Prepare data for template
        sections = []
        for result in analysis:
            section_data = {
                "title": result.name,
                "parameters": result.parameters,
                "summary": self._generate_summary(result.data),
                "table": result.data.to_markdown(index=False),
                "significant_changes": self._get_significant_changes(result.data)
            }
            sections.append(section_data)
        
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "commit_info": commit_info or {},
            "sections": sections,
            "total_sections": len(sections)
        }
        
        with open(output_file, "w") as f:
            f.write(template.render(**report_data))

    def generate_html(self, analysis: List[AnalysisResult], output_file: Path, commit_info: Optional[Dict] = None) -> None:
        """Generate HTML report.

        Args:
            analysis: List of analysis results
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        template = self.env.get_template("report.html.j2")
        
        # Prepare data for template
        sections = []
        for result in analysis:
            section_data = {
                "title": result.name,
                "parameters": result.parameters,
                "summary": self._generate_summary(result.data),
                "table": self._style_analysis_table(result.data),
                "significant_changes": self._get_significant_changes(result.data),
                "charts": {
                    "latency": self._generate_latency_chart_data(result.data),
                    "bandwidth": self._generate_bandwidth_chart_data(result.data) if 'bandwidth_diff_pct' in result.data.columns else None
                }
            }
            sections.append(section_data)
        
        # Prepare overall report data
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "commit_info": commit_info or {},
            "sections": sections,
            "total_sections": len(sections)
        }
        
        with open(output_file, "w") as f:
            f.write(template.render(**report_data))

    def generate_json(self, analysis: List[AnalysisResult], output_file: Path, commit_info: Optional[Dict] = None) -> None:
        """Generate JSON report.

        Args:
            analysis: List of analysis results
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "commit_info": commit_info or {},
            "sections": [
                {
                    "name": result.name,
                    "parameters": result.parameters,
                    "summary": self._generate_summary(result.data),
                    "data": result.data.to_dict(orient="records"),
                    "significant_changes": self._get_significant_changes(result.data)
                }
                for result in analysis
            ]
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
        summary = {
            "total_tests": len(analysis),
            "avg_latency_diff": analysis["latency_diff_pct"].mean(),
            "max_latency_diff": analysis["latency_diff_pct"].max(),
            "min_latency_diff": analysis["latency_diff_pct"].min(),
        }
        
        # Add bandwidth metrics if they exist
        if 'bandwidth_diff_pct' in analysis.columns:
            summary.update({
                "avg_bandwidth_diff": analysis["bandwidth_diff_pct"].mean(),
                "max_bandwidth_diff": analysis["bandwidth_diff_pct"].max(),
                "min_bandwidth_diff": analysis["bandwidth_diff_pct"].min()
            })
            
        return summary

    def _get_significant_changes(self, analysis: pd.DataFrame, threshold: float = 1.0) -> List[Dict[str, Any]]:
        """Get significant changes exceeding threshold.

        Args:
            analysis: Analysis results DataFrame
            threshold: Threshold for significant changes (percentage)

        Returns:
            List of dictionaries containing significant changes
        """
        # Start with latency changes
        significant = analysis[abs(analysis["latency_diff_pct"]) > threshold]
        
        # Add bandwidth changes if they exist
        if 'bandwidth_diff_pct' in analysis.columns:
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
        if 'bandwidth_diff_pct' not in analysis.columns:
            return None
            
        return {
            "msg_sizes": analysis["msg_size"].tolist(),
            "ref_bandwidth": analysis["ref_bandwidth_avg"].tolist(),
            "test_bandwidth": analysis["test_bandwidth_avg"].tolist(),
            "diff_pct": analysis["bandwidth_diff_pct"].tolist()
        } 