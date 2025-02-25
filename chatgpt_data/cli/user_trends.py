"""Command-line interface for ChatGPT user trends analysis."""

import argparse
import os
from pathlib import Path

from chatgpt_data.analysis.user_analysis import UserAnalysis


def main():
    """Run the user trends analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze ChatGPT user engagement trends"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./rawdata",
        help="Directory containing the raw data files (default: ./rawdata)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to save the output files (default: ./data)",
    )
    parser.add_argument(
        "--engagement-report",
        action="store_true",
        help="Generate user engagement level report",
    )
    parser.add_argument(
        "--non-engaged-report",
        action="store_true",
        help="Generate report of users who have never engaged",
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=20,
        help="Minimum number of messages to be considered highly engaged (default: 20)",
    )
    parser.add_argument(
        "--low-threshold",
        type=int,
        default=5,
        help="Maximum number of messages to be considered low engaged (default: 5)",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run the analysis
    analyzer = UserAnalysis(args.data_dir, output_dir)
    
    # Generate trend graphs
    analyzer.generate_all_trends()
    print(f"User trend analysis completed. Results saved to {output_dir}")
    
    # Generate engagement report if requested
    if args.engagement_report:
        engagement_file = output_dir / "user_engagement_report.csv"
        analyzer.generate_engagement_report(output_file=str(engagement_file))
    
    # Generate non-engaged users report if requested
    if args.non_engaged_report:
        non_engaged_file = output_dir / "non_engaged_users_report.csv"
        analyzer.generate_non_engagement_report(output_file=str(non_engaged_file))


if __name__ == "__main__":
    main()
