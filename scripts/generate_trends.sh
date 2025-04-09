#!/bin/bash

# Script to generate complete ChatGPT usage trends in one go
# This script:
#  1. Fetches raw data from the Compliance API
#  2. Processes it into weekly chunks starting from a specific date
#  3. Generates trend analysis and visualizations

set -e  # Exit immediately if a command exits with non-zero status

# Configuration
DATA_DIR="./rawdata"
START_DATE="2024-11-02"
OUTPUT_DIR="./data"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --start-date)
      START_DATE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--data-dir DIR] [--output-dir DIR] [--start-date YYYY-MM-DD]"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

echo "====================================================="
echo "Starting ChatGPT data processing workflow"
echo "====================================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Start date: $START_DATE"
echo "====================================================="

# Step 1: Fetch raw data from the API
echo "Step 1: Fetching raw data from Enterprise Compliance API..."
if ! fetch_raw_data --output-dir "$DATA_DIR"; then
    echo "Error: Failed to fetch raw data from API"
    exit 1
fi
echo "Raw data fetched successfully!"

# Step 2: Process the raw data into engagement metrics with weekly chunks
echo "Step 2: Processing raw data into weekly engagement metrics..."
if ! process_engagement --input-dir "$DATA_DIR" --output-dir "$DATA_DIR" --start-date "$START_DATE" --weekly-chunks; then
    echo "Error: Failed to process engagement data"
    exit 1
fi
echo "Weekly engagement metrics processed successfully!"

# Step 3: Generate all trends and visualizations
echo "Step 3: Generating trend analyses and visualizations..."
if ! all_trends --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR"; then
    echo "Error: Failed to generate trends"
    exit 1
fi
echo "Trend analysis complete!"

echo "====================================================="
echo "ChatGPT data processing workflow complete!"
echo "Raw data and engagement metrics: $DATA_DIR"
echo "Trend visualizations and reports: $OUTPUT_DIR"
echo "====================================================="
