"""CLI tool to fetch raw data via the Enterprise Compliance API."""

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import traceback
import logging

from chatgpt_data.api.compliance_api import EnterpriseComplianceAPI
from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from dotenv import load_dotenv

from chatgpt_data.cli.api_reports import (
    parse_date,
    get_users_conversations,
    save_raw_data,
)

# Configure logger
logger = logging.getLogger('fetch_raw_data')
logger.setLevel(logging.INFO)

# Create console handler if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def main() -> int:
    """Main entry point for the fetch raw data CLI tool."""
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Fetch raw data from the ChatGPT Enterprise Compliance API"
    )
    
    # Data options
    data_group = parser.add_argument_group("Data Options")
    data_group.add_argument(
        "--workspace-id",
        help="Workspace ID to fetch data from (default: from environment variable)",
    )
    data_group.add_argument(
        "--api-key",
        help="Enterprise API key with compliance_export scope (default: from environment variable)",
    )
    data_group.add_argument(
        "--org-id",
        help="Organization ID (default: from environment variable)",
    )
    data_group.add_argument(
        "--output-dir",
        default="./reports",
        help="Directory to save raw data (default: ./reports)",
    )
    data_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed debug logging for verification",
    )
    
    # Add page_size parameter
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Number of items to request per API page (default: 200, max: 200)"
    )
    
    args = parser.parse_args()
    
    # Run the command
    try:
        # Initialize API client
        api_key = args.api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_ENTERPRISE_API_KEY")
        org_id = args.org_id or os.environ.get("ORG_ID") or os.environ.get("OPENAI_ORGANIZATION")
        workspace_id = args.workspace_id or os.environ.get("WORKSPACE_ID")
        
        # Validate required credentials
        missing_credentials = False
        if not api_key:
            logger.error("ERROR: API key is required. Provide via --api-key or API_KEY env var")
            missing_credentials = True
            
        if not org_id:
            logger.warning("WARNING: Organization ID not provided. Using default value")
            org_id = "org_default"
            
        if not workspace_id:
            logger.warning("WARNING: Workspace ID not provided. Using default value")
            workspace_id = "05a09bbb-00b5-4224-bee8-739bb86ec062"
        
        if missing_credentials:
            logger.error("Exiting due to missing credentials")
            return 1
        
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Initialize the API client
            api = EnterpriseComplianceAPI(
                api_key=api_key,
                org_id=org_id,
                workspace_id=workspace_id,
                output_dir=str(output_dir),
                page_size=min(args.page_size, 200),  # Ensure page_size doesn't exceed API limit
            )
            
            # Get user engagement metrics
            logger.info("Fetching users and conversations data...")
            raw_data = get_users_conversations(api, debug_logging=args.debug)

            # Save raw data to disk for later analysis
            pickle_path, json_path = save_raw_data(raw_data, str(output_dir))
            
            logger.info(f"Raw data saved successfully to:")
            logger.info(f"  - {pickle_path} (for Python processing)")
            logger.info(f"  - {json_path} (for external analysis)")
            
            logger.info("\nTo process this data into engagement metrics, run:")
            logger.info(f"  python -m chatgpt_data.cli.process_engagement --input-file {pickle_path}")
            
            return 0
    
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error fetching workspace data: {str(e)}")
            logger.error("This could be due to:")
            logger.error("  - Invalid API credentials")
            logger.error("  - Insufficient permissions for the compliance API")
            logger.error("  - Network connectivity issues")
            logger.error("  - API rate limiting")
            
            logger.info("\nNote: If you're experiencing API rate limiting, try using a smaller page size:")
            logger.info("  python -m chatgpt_data.cli.fetch_raw_data --page-size 50")
            
            return 1
            
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
