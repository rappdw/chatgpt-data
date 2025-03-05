#!/bin/bash

# Script to run get_usage_data with dates derived from existing engagement files
# Usage: ./run_usage_data.sh

# Set the output directory
OUTPUT_DIR="./foo"
RAWDATA_DIR="./rawdata"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find the most recent proofpoint_user_engagement file
LATEST_FILE=$(ls -1 "$RAWDATA_DIR"/proofpoint_user_engagement_*.csv 2>/dev/null | sort -r | head -n 1)

if [ -z "$LATEST_FILE" ]; then
    echo "No existing proofpoint_user_engagement files found in $RAWDATA_DIR"
    exit 1
fi

# Extract date range from filename
# Expected format: proofpoint_user_engagement_YYYYMMDD_YYYYMMDD.csv
FILENAME=$(basename "$LATEST_FILE")
DATE_PART=${FILENAME#proofpoint_user_engagement_}
DATE_PART=${DATE_PART%.csv}

# Split into start and end dates
START_DATE_STR=${DATE_PART%_*}
END_DATE_STR=${DATE_PART#*_}

# Format dates for command line (YYYY-MM-DD)
START_DATE="${START_DATE_STR:0:4}-${START_DATE_STR:4:2}-${START_DATE_STR:6:2}"
END_DATE="${END_DATE_STR:0:4}-${END_DATE_STR:4:2}-${END_DATE_STR:6:2}"

echo "Found date range: $START_DATE to $END_DATE from file $FILENAME"

# Run the get_usage_data command
echo "Running: get_usage_data --output-dir $OUTPUT_DIR --start-date $START_DATE --end-date $END_DATE"
get_usage_data --output-dir "$OUTPUT_DIR" --start-date "$START_DATE" --end-date "$END_DATE"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Successfully generated usage data in $OUTPUT_DIR"
else
    echo "Error running get_usage_data command"
    exit 1
fi
