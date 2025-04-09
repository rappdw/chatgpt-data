"""CLI tool to process raw data into engagement metrics."""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
import traceback
import logging
import pickle
import glob
import sys

from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from dotenv import load_dotenv

from chatgpt_data.cli.api_reports import (
    parse_date,
    RawData,
    process_engagement_data,
    process_data_in_weekly_chunks,
    apply_date_filters,
    load_existing_data,
)

# Configure logger
logger = logging.getLogger('process_engagement')
logger.setLevel(logging.INFO)

# Create console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def load_raw_data(input_file: str) -> Optional[RawData]:
    """
    Load raw data from a pickle file.
    
    Args:
        input_file: Path to the pickle file to load
        
    Returns:
        RawData object or None if loading failed
    """
    try:
        with open(input_file, 'rb') as f:
            raw_data = pickle.load(f)
            logger.info(f"Loaded raw data from {input_file}")
            return raw_data
    except Exception as e:
        logger.error(f"Error loading raw data: {str(e)}")
        return None


def find_latest_raw_data(data_dir: str) -> Optional[str]:
    """
    Find the latest raw data pickle file in the given directory.
    
    Args:
        data_dir: Directory to search for pickle files
        
    Returns:
        Path to the latest pickle file or None if no files found
    """
    pickle_files = glob.glob(os.path.join(data_dir, "raw_data_*.pkl"))
    if not pickle_files:
        return None
    
    # Sort by modification time, newest first
    pickle_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return pickle_files[0]


def find_existing_csv_files(output_dir: str) -> List[str]:
    """
    Find existing CSV files in the output directory that match engagement report patterns.
    
    Args:
        output_dir: Directory to search for CSV files
        
    Returns:
        List of paths to existing CSV files
    """
    patterns = [
        "user_engagement_*.csv",
        "non_engagement_*.csv",
        "gpt_engagement_*.csv",
        "gpt_non_engagement_*.csv"
    ]
    
    existing_files = []
    for pattern in patterns:
        existing_files.extend(glob.glob(os.path.join(output_dir, pattern)))
    
    return existing_files


def main() -> int:
    """Main entry point for the process engagement CLI tool."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Process raw data into engagement metrics"
    )
    
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--input-file",
        help="Path to raw data pickle file to process",
    )
    data_group.add_argument(
        "--input-dir",
        default="./rawdata",
        help="Directory to search for raw data files (default: ./rawdata)",
    )
    data_group.add_argument(
        "--output-dir",
        default="./rawdata",
        help="Directory to save processed reports (default: ./rawdata)",
    )
    data_group.add_argument(
        "--start-date",
        type=parse_date,
        help="Start date for reports (YYYY-MM-DD, default: earliest observed date)",
    )
    data_group.add_argument(
        "--end-date",
        type=parse_date,
        help="End date for reports (YYYY-MM-DD, default: most recent observed date)",
    )
    data_group.add_argument(
        "--weekly-chunks",
        action="store_true",
        help="Process data in weekly chunks instead of the full date range",
    )
    data_group.add_argument(
        "--allow-partial-weeks",
        action="store_true",
        help="Process partial weeks at the end of the date range",
    )
    
    args = parser.parse_args()
    
    # Run the command
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load raw data
        raw_data = None
        
        if args.input_file:
            # Load from specific file
            raw_data = load_raw_data(args.input_file)
            if not raw_data:
                logger.error(f"Failed to load raw data from {args.input_file}")
                return 1
        else:
            # Find latest file in input directory
            latest_file = find_latest_raw_data(args.input_dir)
            if latest_file:
                logger.info(f"Found latest raw data file: {latest_file}")
                raw_data = load_raw_data(latest_file)
                if not raw_data:
                    logger.error(f"Failed to load raw data from {latest_file}")
                    return 1
            else:
                logger.error(f"No raw data files found in {args.input_dir}")
                logger.info("Please run fetch_raw_data.py first to generate raw data files.")
                return 1
        
        # Apply date filters if provided
        raw_data = apply_date_filters(raw_data, args.start_date, args.end_date)
        
        # Print date range
        earliest_timestamp = raw_data.earliest_message_timestamp
        latest_timestamp = raw_data.latest_message_timestamp
        
        earliest_date = datetime.fromtimestamp(earliest_timestamp, DEFAULT_TIMEZONE)
        latest_date = datetime.fromtimestamp(latest_timestamp, DEFAULT_TIMEZONE)
        
        logger.info(f"Processing data from {earliest_date.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")
        
        # Check for existing CSV files
        existing_files = find_existing_csv_files(str(output_dir))
        if existing_files:
            logger.warning(f"Found {len(existing_files)} existing CSV files in {output_dir}")
            for file in existing_files[:5]:  # Show up to 5 files
                logger.warning(f"  - {os.path.basename(file)}")
            if len(existing_files) > 5:
                logger.warning(f"  - ... and {len(existing_files) - 5} more")
                
            # Ask user for confirmation
            response = input("\nDelete these files before proceeding? [y/N] ").strip().lower()
            if response == 'y' or response == 'yes':
                for file in existing_files:
                    try:
                        os.remove(file)
                        logger.info(f"Deleted: {os.path.basename(file)}")
                    except Exception as e:
                        logger.error(f"Error deleting {file}: {str(e)}")
                logger.info(f"Deleted {len(existing_files)} existing files")
            else:
                logger.info("Keeping existing files")
        
        try:
            if args.weekly_chunks:
                # Process data in weekly chunks
                logger.info("Processing data in weekly chunks...")
                process_data_in_weekly_chunks(raw_data, str(output_dir), allow_partial_weeks=args.allow_partial_weeks)
            else:
                # Process all data at once
                logger.info("Processing all data at once...")
                process_engagement_data(raw_data, str(output_dir))
                
            logger.info(f"Engagement metrics saved to {output_dir}")
            return 0
            
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error processing data: {str(e)}")
            return 1
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
