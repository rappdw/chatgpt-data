"""CLI tool to fetch ChatGPT usage reports via the Enterprise Compliance API."""

import argparse
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union

from chatgpt_data.api.compliance_api import EnterpriseComplianceAPI
from chatgpt_data.cli.all_trends import main as run_all_trends


def parse_date(date_str: str) -> datetime:
    """Parse a date string in YYYY-MM-DD format.

    Args:
        date_str: Date string to parse

    Returns:
        Parsed datetime object
    """
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def get_default_dates() -> tuple[datetime, datetime]:
    """Get default date range (previous week).

    Returns:
        Tuple of (start_date, end_date)
    """
    today = datetime.now()
    # Start from last Saturday
    days_since_saturday = (today.weekday() + 2) % 7
    end_date = today - timedelta(days=days_since_saturday)
    start_date = end_date - timedelta(days=6)  # Previous Sunday to Saturday
    return start_date, end_date


def main() -> None:
    """Main entry point for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="Fetch ChatGPT usage reports via the Enterprise Compliance API"
    )
    
    # Data options
    parser.add_argument(
        "--api-key", 
        help="Enterprise API key (defaults to OPENAI_ENTERPRISE_API_KEY env var)"
    )
    parser.add_argument(
        "--org-id", 
        help="Organization ID (defaults to OPENAI_ORG_ID env var)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./rawdata",
        help="Directory to save downloaded reports (default: ./rawdata)"
    )
    
    # Date range options
    default_start, default_end = get_default_dates()
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=default_start,
        help=f"Start date (YYYY-MM-DD) (default: {default_start.strftime('%Y-%m-%d')})"
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=default_end,
        help=f"End date (YYYY-MM-DD) (default: {default_end.strftime('%Y-%m-%d')})"
    )
    
    # Report options
    parser.add_argument(
        "--skip-user-report",
        action="store_true",
        help="Skip downloading user engagement report"
    )
    parser.add_argument(
        "--skip-gpt-report",
        action="store_true",
        help="Skip downloading GPT engagement report"
    )
    
    # Analysis options
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run data analysis after downloading reports"
    )
    parser.add_argument(
        "--analysis-output-dir",
        default="./data",
        help="Directory to save analysis output (default: ./data)"
    )
    
    args = parser.parse_args()
    
    # Initialize API client
    api = EnterpriseComplianceAPI(
        api_key=args.api_key,
        org_id=args.org_id,
        output_dir=args.output_dir
    )
    
    downloaded_files = []
    
    # Download reports
    if not args.skip_user_report:
        try:
            user_report_path = api.get_user_engagement_report(
                start_date=args.start_date,
                end_date=args.end_date
            )
            print(f"Downloaded user engagement report: {user_report_path}")
            downloaded_files.append(user_report_path)
        except Exception as e:
            print(f"Error downloading user engagement report: {str(e)}")
    
    if not args.skip_gpt_report:
        try:
            gpt_report_path = api.get_gpt_engagement_report(
                start_date=args.start_date,
                end_date=args.end_date
            )
            print(f"Downloaded GPT engagement report: {gpt_report_path}")
            downloaded_files.append(gpt_report_path)
        except Exception as e:
            print(f"Error downloading GPT engagement report: {str(e)}")
    
    # Run analysis if requested
    if args.run_analysis and downloaded_files:
        print("Running data analysis...")
        try:
            # Use modified sys.argv to call all_trends with the right parameters
            import sys
            original_argv = sys.argv.copy()
            sys.argv = [
                "all_trends",
                "--data-dir", args.output_dir,
                "--output-dir", args.analysis_output_dir
            ]
            
            run_all_trends()
            
            # Restore original argv
            sys.argv = original_argv
            
            print(f"Analysis complete. Results saved to {args.analysis_output_dir}")
        except Exception as e:
            print(f"Error running analysis: {str(e)}")


if __name__ == "__main__":
    main()