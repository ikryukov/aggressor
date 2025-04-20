# HPC Performance Monitor

A Python service for detecting performance regressions in HPC middleware through automated benchmarking and analysis.

## Features

- **Git Integration**: Automated checkout and building of specific commits
- **Flexible Benchmarking**: Support for various HPC benchmarks with configurable parameters
- **Slurm Support**: Run benchmarks locally or through Slurm job scheduler
- **Comprehensive Analysis**: Statistical analysis of performance metrics with regression detection
- **Rich Reporting**: Generate reports in Markdown, HTML, or JSON formats

## Requirements

- Python 3.10 or later
- Git
- MPI implementation (e.g., OpenMPI, MPICH)
- Slurm (optional, for cluster deployment)
- CUDA/ROCm (optional, for GPU support)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hpc-perf-monitor.git
   cd hpc-perf-monitor
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/macOS
   # or
   .\venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

## Configuration

Create a YAML configuration file (see `config.example.yaml` for a template) with the following sections:

- **Git Repository**: URL and commit information
- **Build Settings**: Configure flags, make options, and environment variables
- **Benchmarks**: List of benchmarks with their parameters
- **Regression Analysis**: Thresholds and metric weights
- **Report Formats**: Multiple output format selection

Example:
```yaml
repo_url: "https://github.com/openucx/ucc.git"
work_dir: "/tmp/hpc-perf-monitor"
ref_commit: "master"
test_commits:
  - "HEAD~1"
  - "HEAD"

build:
  source_dir: "/tmp/hpc-perf-monitor/source"
  build_dir: "/tmp/hpc-perf-monitor/build"
  configure_flags:
    - "--enable-debug"
    - "--with-cuda=/usr/local/cuda"
  # ... more settings ...

report_formats:
  - "markdown"
  - "html"
  - "json"
```

## Usage

Run the performance analysis:
```bash
python -m hpc_perf_monitor config.yaml
```

Options:
- `--output-dir`: Directory to store results (default: `results`)
- `--debug`: Enable debug logging

## Report Formats

### Markdown
- Simple text-based format
- Great for version control and GitHub integration
- Easy to read and edit

### HTML
- Rich interactive format
- Includes styling and formatting
- Ideal for sharing with stakeholders

### JSON
- Machine-readable format
- Suitable for further processing
- Can be integrated into CI/CD pipelines

## Extending the Tool

### Adding New Benchmarks
1. Create a new benchmark configuration in your YAML file
2. Implement a parser for the benchmark output
3. Register the parser in the plugin system

### Custom Metrics
1. Add new metric definitions to your configuration
2. Update the regression analysis thresholds
3. Modify the report templates if needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.