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

### get_usage_data
A command-line tool that fetches ChatGPT usage data via the Enterprise Compliance API and saves it to the rawdata directory.

```bash
get_usage_data [--api-key API_KEY] [--org-id ORG_ID] [--output-dir DIRECTORY] [--start-date DATE] [--end-date DATE] [--skip-user-report] [--skip-gpt-report] [--run-analysis] [--analysis-output-dir DIRECTORY]
```

Options:
- `--api-key`: Enterprise API key (defaults to OPENAI_ENTERPRISE_API_KEY env var)
- `--org-id`: Organization ID (defaults to OPENAI_ORG_ID env var)
- `--output-dir`: Directory to save downloaded reports (default: ./rawdata)
- `--start-date`: Start date in YYYY-MM-DD format (default: previous Sunday)
- `--end-date`: End date in YYYY-MM-DD format (default: previous Saturday)
- `--skip-user-report`: Skip downloading user engagement report
- `--skip-gpt-report`: Skip downloading GPT engagement report
- `--run-analysis`: Run data analysis after downloading reports
- `--analysis-output-dir`: Directory to save analysis output (default: ./data)

### user_trends
A command-line tool that provides analysis of ChatGPT user trends. It generates trend graphs (PNG files) in the data directory.

```bash
user_trends [--data-dir DIRECTORY] [--output-dir DIRECTORY] [--engagement-report] [--non-engaged-report] [--high-threshold N] [--low-threshold N] [--latest-period-only]
```

Options:
- `--data-dir`: Directory containing the raw data files (default: ./rawdata)
- `--output-dir`: Directory to save the output files (default: ./data)
- `--engagement-report`: Generate a report of user engagement levels based on average message count
- `--non-engaged-report`: Generate a report of users who have never engaged
- `--high-threshold`: Minimum average number of messages to be considered highly engaged (default: 20)
- `--low-threshold`: Maximum average number of messages to be considered low engaged (default: 5)
- `--latest-period-only`: For non-engaged report, only consider the latest period instead of all periods

### gpt_trends
A command-line tool that provides analysis of ChatGPT custom GPT trends. It generates trend graphs (PNG files) in the data directory.

```bash
gpt_trends [--data-dir DIRECTORY] [--output-dir DIRECTORY]
```

Options:
- `--data-dir`: Directory containing the raw data files (default: ./rawdata)
- `--output-dir`: Directory to save the output files (default: ./data)

### all_trends
A comprehensive command-line tool that runs both user and GPT analyses in one go. It generates all trend graphs and reports in the data directory.

```bash
all_trends [--data-dir DIRECTORY] [--output-dir DIRECTORY] [--skip-user-trends] [--skip-gpt-trends] [--skip-engagement-report] [--skip-non-engaged-report] [--only-message-histogram] [--histogram-bins N] [--histogram-max N] [--latest-period-only] [--high-threshold N] [--low-threshold N]
```

Options:
- `--data-dir`: Directory containing the raw data files (default: ./rawdata)
- `--output-dir`: Directory to save the output files (default: ./data)
- `--skip-user-trends`: Skip generating user trend graphs
- `--skip-gpt-trends`: Skip generating GPT trend graphs
- `--skip-engagement-report`: Skip generating user engagement level report
- `--skip-non-engaged-report`: Skip generating report of non-engaged users
- `--only-message-histogram`: Only generate the message histogram
- `--histogram-bins`: Number of bins for the message histogram (default: 20)
- `--histogram-max`: Maximum value to include in the message histogram (default: no limit)
- `--latest-period-only`: For non-engaged report, only consider the latest period instead of all periods
- `--high-threshold`: Minimum average number of messages to be considered highly engaged (default: 20)
- `--low-threshold`: Maximum average number of messages to be considered low engaged (default: 5)

## Data Preparation for Analysis

Before running the analysis tools, you need to prepare the necessary data files in the `rawdata` directory:

### 1. ChatGPT Usage Reports
Obtain the latest ChatGPT usage reports from OpenAI:
- Download the report attachments from the weekly email sent by OpenAI
- Save the user engagement files (prefixed with "proofpoint_user_engagement") to the `rawdata` directory
- Save the GPT engagement files (prefixed with "proofpoint_gpt_engagement") to the `rawdata` directory

Alternatively, you can use the `get_usage_data` command to fetch these reports directly via the Enterprise Compliance API.

### 2. Active Directory Export
To resolve user names and email addresses:
- Export user data from Active Directory
- Save the file as `AD_export.csv` in the `rawdata` directory
- Ensure the file contains at least the following columns:
  - `userPrincipalName` (email address)
  - `displayName` (full name)
  - `mail` (alternative email address, if available)

### 3. Management Chain Information
To include organizational hierarchy in the reports:
- Export company information from Workday
- Process this data using the `management_chains` tool from the `graphviz_workday` repository
- Save the resulting `management_chains.json` file in the `rawdata` directory
- The JSON file should map employee names to their management chain (array of manager names)

Example `management_chains.json` format:
```json
{
  "Employee Name": [
    "Direct Manager",
    "Manager's Manager",
    "Executive",
    "CEO"
  ]
}
```

### Running the Analysis
Once all data files are in place, run the analysis:

```bash
# Run the complete analysis
python -m chatgpt_data.cli.all_trends

# Generate only the engagement report with management chain information
python -m chatgpt_data.cli.all_trends --skip-gpt-trends --skip-non-engaged-report
```

The analysis will:
1. Load and process all data files
2. Resolve user names from the AD export
3. Add management chain information from the JSON file
4. Generate trend graphs and reports
5. Save all outputs to the specified output directory (default: `./data`)

## Generated Graphs

### User Trends
- `active_users_trend.png`: Trend of active users over time
- `message_volume_trend.png`: Trend of message volume over time
- `gpt_usage_trend.png`: Trend of GPT usage by users over time
- `message_histogram.png`: Distribution of average messages sent by users

### GPT Trends
- `active_gpts_trend.png`: Trend of active custom GPTs over time
- `gpt_messages_trend.png`: Trend of messages sent to custom GPTs over time
- `unique_messagers_trend.png`: Trend of unique users messaging custom GPTs over time

## Generated Reports

### User Engagement Reports
- `user_engagement_report.csv`: Report of user engagement levels (high, medium, low, none) based on average message count across all periods
  - Includes active period percentage (active periods / eligible periods since user creation)
- `non_engaged_users_report.csv`: Report of users who have never engaged (either across all tracked periods or only in the latest period, depending on the `--latest-period-only` option)

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
