"""Report generation functionality."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
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
            'ref_latency_avg': 'Reference Latency Avg (us)',
            'test_latency_avg': 'Test Latency Avg (us)',
            'latency_avg_diff_pct': 'Latency Avg Difference (%)',
            'ref_bandwidth_avg': 'Reference Bandwidth Avg (GB/s)',
            'test_bandwidth_avg': 'Test Bandwidth Avg (GB/s)',
            'bandwidth_diff_pct': 'Bandwidth Difference (%)',
            'ref_latency_min': 'Reference Latency Min (us)',
            'test_latency_min': 'Test Latency Min (us)',
            'latency_min_diff_pct': 'Latency Min Difference (%)',
            'ref_latency_max': 'Reference Latency Max (us)',
            'test_latency_max': 'Test Latency Max (us)',
            'latency_max_diff_pct': 'Latency Max Difference (%)'
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
        analysis: list[AnalysisResult],
        output_file: Path,
        format: str = "markdown",
        commit_info: dict | None = None
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

    def generate_markdown(self, analysis: list[AnalysisResult], output_file: Path, commit_info: dict | None = None) -> None:
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
                "significant_changes": result.significant_changes,
                "has_regression": result.has_regression
            }
            sections.append(section_data)
        
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "commit_info": commit_info or {},
            "sections": sections,
            "total_sections": len(sections),
            "has_regression": any(result.has_regression for result in analysis)
        }
        
        with open(output_file, "w") as f:
            f.write(template.render(**report_data))

    def generate_html(self, analysis: list[AnalysisResult], output_file: Path, commit_info: dict | None = None) -> None:
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
                "significant_changes": result.significant_changes,
                "has_regression": result.has_regression,
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
            "total_sections": len(sections),
            "has_regression": any(result.has_regression for result in analysis)
        }
        
        with open(output_file, "w") as f:
            f.write(template.render(**report_data))

    def generate_json(self, analysis: list[AnalysisResult], output_file: Path, commit_info: dict | None = None) -> None:
        """Generate JSON report.

        Args:
            analysis: List of analysis results
            output_file: Path to save report to
            commit_info: Optional dictionary with commit information
        """
        # Extract AI analysis if available without including it in the main commit info
        ai_analysis = None
        if commit_info and 'ai_analysis' in commit_info:
            ai_analysis = commit_info['ai_analysis']
            # Create a copy of commit_info without the ai_analysis
            commit_info_without_ai = {k: v for k, v in commit_info.items() if k != 'ai_analysis'}
        else:
            commit_info_without_ai = commit_info or {}
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "commit_info": commit_info_without_ai,
            "sections": [
                {
                    "name": result.name,
                    "parameters": result.parameters,
                    "summary": self._generate_summary(result.data),
                    "data": result.data.to_dict(orient="records"),
                    "significant_changes": result.significant_changes,
                    "has_regression": result.has_regression
                }
                for result in analysis
            ],
            "has_regression": any(result.has_regression for result in analysis)
        }
        
        # Add AI analysis if available
        if ai_analysis:
            report_data["ai_analysis"] = {
                "content": ai_analysis["content"],
                "file": ai_analysis["file"],
            }
            
            # Add benchmark info if available
            if "benchmark_info" in ai_analysis:
                report_data["ai_analysis"]["benchmark_info"] = ai_analysis["benchmark_info"]
        
        with open(output_file, "w") as f:
            json.dump(report_data, f, indent=2)

    def _generate_summary(self, analysis: pd.DataFrame) -> dict[str, Any]:
        """Generate summary statistics from analysis data.

        Args:
            analysis: Analysis results DataFrame

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            "total_tests": len(analysis),
            "avg_latency_diff": analysis["latency_avg_diff_pct"].mean(),
            "max_latency_diff": analysis["latency_max_diff_pct"].max(),
            "min_latency_diff": analysis["latency_min_diff_pct"].min(),
        }
        
        # Add bandwidth metrics if they exist
        if 'bandwidth_diff_pct' in analysis.columns:
            summary.update({
                "avg_bandwidth_diff": analysis["bandwidth_diff_pct"].mean(),
                "max_bandwidth_diff": analysis["bandwidth_diff_pct"].max(),
                "min_bandwidth_diff": analysis["bandwidth_diff_pct"].min()
            })
            
        return summary

    def _get_significant_changes(self, analysis: pd.DataFrame, threshold: float = 1.0) -> list[dict[str, Any]]:
        """Get significant changes exceeding threshold.

        Args:
            analysis: Analysis results DataFrame
            threshold: Threshold for significant changes (percentage)

        Returns:
            List of dictionaries containing significant changes
        """
        # Start with latency changes
        significant = analysis[abs(analysis["latency_avg_diff_pct"]) > threshold]
        
        # Add bandwidth changes if they exist
        if 'bandwidth_diff_pct' in analysis.columns:
            significant = analysis[
                (abs(analysis["latency_avg_diff_pct"]) > threshold) |
                (abs(analysis["bandwidth_diff_pct"]) > threshold)
            ]
        
        return significant.to_dict(orient="records")

    def _generate_latency_chart_data(self, analysis: pd.DataFrame) -> dict[str, list[float]]:
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
            "diff_pct": analysis["latency_avg_diff_pct"].tolist()
        }

    def _generate_bandwidth_chart_data(self, analysis: pd.DataFrame) -> dict[str, list[float]]:
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