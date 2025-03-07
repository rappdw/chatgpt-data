# ChatGPT Data Analysis 1.1.0

We're pleased to announce the release of ChatGPT Data Analysis 1.1.0, featuring significant improvements to the Enterprise Compliance API client and enhanced documentation.

## What's New

### Enhanced API Client

- **Improved Error Handling**: Graceful handling for API authorization errors (401/403)
- **Comprehensive Logging**: Better logging for API consistency issues
- **Pagination Support**: Robust support for pagination in the conversations API endpoint
- **Enhanced Testing**: More comprehensive mock data generation for testing scenarios
- **Better Error Messages**: Improved CLI with detailed error messages and credential validation
- **Consistent Test Mode**: Standardized workspace ID usage for test mode

### Web Interface Documentation

- Added documentation for viewing reports through the web interface
- Instructions for starting the server and accessing the dashboard
- Overview of interactive features available in the web UI

### Other Improvements

- Clarified API credential requirements in documentation
- Updated CLI command examples
- Various code quality improvements

## Installation

```bash
# Using uv (recommended)
uv pip install chatgpt-data

# Using pip
pip install chatgpt-data
```

---

# ChatGPT Data Analysis 1.0.0

We're excited to announce the first stable release of the ChatGPT Data Analysis package!

## What's New

### Comprehensive Analysis CLI

The highlight of this release is the new `all_trends` CLI tool that combines user and GPT trend analyses in one convenient command. This tool provides:

- Unified analysis of both user engagement and GPT usage data
- Granular control via command-line options
- Comprehensive reports and trend graphs
- Flexible configuration for engagement thresholds

### Key Features

- **User Analysis**: Track active users, message volume, and GPT usage over time
- **GPT Analysis**: Monitor active GPTs, message volume to GPTs, and unique messagers
- **Engagement Reports**: Identify high, medium, low, and non-engaged users
- **Non-Engagement Reports**: Find users who have never engaged with the platform

### Installation

```bash
# Using uv (recommended)
uv pip install chatgpt-data

# Using pip
pip install chatgpt-data
```

### Getting Started

```bash
# Run a comprehensive analysis with default settings
all_trends

# Customize your analysis
all_trends --data-dir /path/to/data --output-dir /path/to/output --high-threshold 25 --low-threshold 3
```

See the [README.md](https://github.com/proofpoint/chatgpt-data/blob/main/README.md) for complete documentation on all available options and features.

## Acknowledgements

Thank you to everyone who contributed to making this release possible!
