# ChatGPT Usage Data

## Overview
A Python package that provides analysis of ChatGPT usage data. The package analyzes user engagement and custom GPT usage data to generate trend graphs.

## Data Structure
The `rawdata` directory contains usage data from users and custom GPTs:
- User data files are prefixed by "proofpoint_user_engagement"
- Custom GPT data files are prefixed by "proofpoint_gpt_engagement"
- "glossary_gpt_engagement.csv" provides detailed descriptions of the columns in the GPT engagement data
- "glossary_user_engagement.csv" provides detailed descriptions of the columns in the user engagement data

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

## CLI Commands

### user_trends
A command-line tool that provides analysis of ChatGPT user trends. It generates trend graphs (PNG files) in the data directory.

```bash
user_trends [--data-dir DIRECTORY] [--output-dir DIRECTORY] [--engagement-report] [--non-engaged-report] [--high-threshold N] [--low-threshold N]
```

Options:
- `--data-dir`: Directory containing the raw data files (default: ./rawdata)
- `--output-dir`: Directory to save the output files (default: ./data)
- `--engagement-report`: Generate a report of user engagement levels based on average message count
- `--non-engaged-report`: Generate a report of users who have never engaged
- `--high-threshold`: Minimum average number of messages to be considered highly engaged (default: 20)
- `--low-threshold`: Maximum average number of messages to be considered low engaged (default: 5)

### gpt_trends
A command-line tool that provides analysis of ChatGPT custom GPT trends. It generates trend graphs (PNG files) in the data directory.

```bash
gpt_trends [--data-dir DIRECTORY] [--output-dir DIRECTORY]
```

Options:
- `--data-dir`: Directory containing the raw data files (default: ./rawdata)
- `--output-dir`: Directory to save the output files (default: ./data)

## Generated Graphs

### User Trends
- `active_users_trend.png`: Trend of active users over time
- `message_volume_trend.png`: Trend of message volume over time
- `gpt_usage_trend.png`: Trend of GPT usage by users over time

### GPT Trends
- `active_gpts_trend.png`: Trend of active custom GPTs over time
- `gpt_messages_trend.png`: Trend of messages sent to custom GPTs over time
- `unique_messagers_trend.png`: Trend of unique users messaging custom GPTs over time

## Generated Reports

### User Engagement Reports
- `user_engagement_report.csv`: Report of user engagement levels (high, medium, low, none) based on average message count across all periods
- `non_engaged_users_report.csv`: Report of users who have never engaged across all tracked periods

## Development

### Setup Development Environment
```bash
# Create and activate a virtual environment using uv
uv venv

# Install development dependencies
uv pip install -e ".[dev]"
```

### Run Tests
```bash
pytest
```

### Code Quality
This project uses:
- black for code formatting
- isort for import sorting
- ruff for code quality checks
- mypy for type checking
- flake8 for code style checks

### Continuous Integration
This project uses GitHub Actions for continuous integration. The workflow includes:
- Linting with black, isort, ruff, mypy, and flake8
- Running tests with pytest
- Building the package