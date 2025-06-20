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

### fetch_raw_data
A command-line tool that fetches raw data from the ChatGPT Enterprise Compliance API and saves it for later processing.

```bash
fetch_raw_data [--workspace-id WORKSPACE_ID] [--api-key API_KEY] [--org-id ORG_ID] [--output-dir DIRECTORY] [--debug] [--page-size SIZE]
```

> **Note:** API credentials (API_KEY, ORG_ID) must be provided either as command-line arguments, in a `.env` file in the project root, or as environment variables.

Options:
- `--workspace-id`: Workspace ID to fetch data from (default: from environment variable)
- `--api-key`: Enterprise API key with compliance_export scope (default: from environment variable)
- `--org-id`: Organization ID (default: from environment variable)
- `--output-dir`: Directory to save raw data (default: ./reports)
- `--debug`: Enable detailed debug logging for verification
- `--page-size`: Number of items to request per API page (default: 200, max: 200)

### process_engagement
A command-line tool that processes raw data into engagement metrics.

```bash
process_engagement [--input-file FILE] [--input-dir DIRECTORY] [--output-dir DIRECTORY] [--start-date DATE] [--end-date DATE] [--weekly-chunks]
```

Options:
- `--input-file`: Path to raw data pickle file to process
- `--input-dir`: Directory to search for raw data files (default: ./reports)
- `--output-dir`: Directory to save processed reports (default: ./reports)
- `--start-date`: Start date for reports (YYYY-MM-DD, default: earliest observed date)
- `--end-date`: End date for reports (YYYY-MM-DD, default: most recent observed date)
- `--weekly-chunks`: Process data in weekly chunks instead of the full date range

### all_trends
A comprehensive command-line tool that provides analysis of both ChatGPT user and GPT engagement trends. It generates trend graphs and detailed reports to help understand usage patterns.

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

Alternatively, you can use the `fetch_raw_data` command to fetch these reports directly via the Enterprise Compliance API.

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
- Export company information from Workday (select CEO's team and print the team including all levels as excel)
- Save "Proofpoint.xlsx" in the `rawdata` directory
- Process this data using the `management_chains` tool
- This will save the resulting `management_chains.json` file in the `rawdata` directory
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
all_trends
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

## Viewing Reports in the Web Interface

The package includes a web interface that displays the generated graphs and reports in an interactive dashboard.

### Starting the Web Server

To start the web server, run the following command from the project root directory:

```bash
python server.py

# or

./server.py
```

This will start a local web server on port 8000. You should see a message indicating that the server is running:

```
Serving at http://localhost:8000
```

### Accessing the Dashboard

Open your web browser and navigate to:

```
http://localhost:8000
```

The dashboard provides:

1. A visual display of all generated trend graphs
2. Interactive tables for the user engagement reports with:
   - Filtering capabilities
   - Sorting by any column
   - Pagination for large datasets
   - Management chain filtering

### Generating New Reports

You can generate new reports directly from the web interface by clicking the "Generate Reports" button. This will run the `all_trends` command and update the displayed graphs and reports when complete.

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
