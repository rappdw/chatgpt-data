#!/bin/bash

# Script to run get_usage_data for all date ranges found in existing engagement files
# Usage: ./run_all_usage_data.sh [--force]

# Set the output directory
OUTPUT_DIR="./foo"
RAWDATA_DIR="./rawdata"
FORCE=false

# Check for force flag
if [ "$1" = "--force" ]; then
    FORCE=true
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Find all proofpoint_user_engagement files
FILES=$(ls -1 "$RAWDATA_DIR"/proofpoint_user_engagement_*.csv 2>/dev/null | sort)

if [ -z "$FILES" ]; then
    echo "No existing proofpoint_user_engagement files found in $RAWDATA_DIR"
    exit 1
fi

echo "Found $(echo "$FILES" | wc -l | tr -d ' ') engagement files"

# Process each file
for FILE in $FILES; do
    # Extract date range from filename
    # Expected format: proofpoint_user_engagement_YYYYMMDD_YYYYMMDD.csv
    FILENAME=$(basename "$FILE")
    DATE_PART=${FILENAME#proofpoint_user_engagement_}
    DATE_PART=${DATE_PART%.csv}

    # Split into start and end dates
    START_DATE_STR=${DATE_PART%_*}
    END_DATE_STR=${DATE_PART#*_}

    # Format dates for command line (YYYY-MM-DD)
    START_DATE="${START_DATE_STR:0:4}-${START_DATE_STR:4:2}-${START_DATE_STR:6:2}"
    END_DATE="${END_DATE_STR:0:4}-${END_DATE_STR:4:2}-${END_DATE_STR:6:2}"

    # Check if output file already exists
    OUTPUT_FILE="$OUTPUT_DIR/user_engagement_${START_DATE_STR}_${END_DATE_STR}.csv"
    if [ -f "$OUTPUT_FILE" ] && [ "$FORCE" = false ]; then
        echo "Skipping $START_DATE to $END_DATE (output file already exists)"
        continue
    fi

    echo "Processing date range: $START_DATE to $END_DATE from file $FILENAME"

    # Run the get_usage_data command
    echo "Running: get_usage_data --output-dir $OUTPUT_DIR --start-date $START_DATE --end-date $END_DATE"
    get_usage_data --output-dir "$OUTPUT_DIR" --start-date "$START_DATE" --end-date "$END_DATE"

    # Check if the command was successful
    if [ $? -eq 0 ]; then
        echo "Successfully generated usage data for $START_DATE to $END_DATE"
    else
        echo "Error running get_usage_data command for $START_DATE to $END_DATE"
    fi
    
    echo "-----------------------------------"
done

echo "All date ranges processed. Results saved to $OUTPUT_DIR"
