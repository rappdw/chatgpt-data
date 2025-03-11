import argparse
import os
from pathlib import Path

from chatgpt_data.analysis.user_analyzer import UserAnalyzer
from chatgpt_data.analysis.gpt_analyzer import GPTAnalyzer
from chatgpt_data.analysis.report_generator import ReportGenerator


def main():
    """Run a comprehensive analysis of both user and GPT trends."""
    parser = argparse.ArgumentParser(
        description="Analyze ChatGPT user and GPT engagement trends in one go"
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
        "--skip-user-trends",
        action="store_true",
        help="Skip generating user trend graphs",
    )
    parser.add_argument(
        "--skip-gpt-trends",
        action="store_true",
        help="Skip generating GPT trend graphs",
    )
    parser.add_argument(
        "--skip-engagement-report",
        action="store_true",
        help="Skip generating user engagement level report",
    )
    parser.add_argument(
        "--skip-non-engaged-report",
        action="store_true",
        help="Skip generating report of non-engaged users",
    )
    parser.add_argument(
        "--only-message-histogram",
        action="store_true",
        help="Only generate the message histogram",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=20,
        help="Number of bins for the message histogram (default: 20)",
    )
    parser.add_argument(
        "--histogram-max",
        type=int,
        default=None,
        help="Maximum value to include in the message histogram (default: no limit)",
    )
    parser.add_argument(
        "--latest-period-only",
        action="store_true",
        help="For non-engaged report, only consider the latest period instead of all periods",
    )
    parser.add_argument(
        "--high-threshold",
        type=int,
        default=20,
        help="Minimum average number of messages to be considered highly engaged (default: 20)",
    )
    parser.add_argument(
        "--low-threshold",
        type=int,
        default=5,
        help="Maximum average number of messages to be considered low engaged (default: 5)",
    )
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("Starting comprehensive ChatGPT data analysis...")
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # If only message histogram is requested, generate just that
    if args.only_message_histogram:
        print("\n=== Generating Message Histogram ===")
        try:
            user_analyzer = UserAnalyzer(args.data_dir, output_dir)
            user_analyzer.generate_message_histogram(bins=args.histogram_bins, max_value=args.histogram_max)
            print(f"✓ Message histogram saved to {output_dir / 'message_histogram.png'}")
        except Exception as e:
            print(f"Error generating message histogram: {str(e)}")
        
        print("\n=== Analysis Complete ===")
        print(f"Results saved to {output_dir}")
        return
    
    # Run user analysis
    if not args.skip_user_trends:
        print("\n=== Running User Analysis ===")
        try:
            # Create the analyzer
            user_analyzer = UserAnalyzer(args.data_dir, output_dir)
            
            # Generate user trend graphs
            user_analyzer.generate_all_trends()
            print("✓ User trend graphs generated")
            
            # Create report generator
            report_generator = ReportGenerator(user_analyzer)
            
            # Generate engagement report if not skipped
            if not args.skip_engagement_report:
                engagement_file = output_dir / "user_engagement_report.csv"
                report_generator.generate_engagement_report(output_file=str(engagement_file))
                print(f"✓ User engagement report saved to {engagement_file}")
            
            # Generate non-engaged users report if not skipped
            if not args.skip_non_engaged_report:
                non_engaged_file = output_dir / "non_engaged_users_report.csv"
                report_generator.generate_non_engagement_report(
                    output_file=str(non_engaged_file),
                    only_latest_period=args.latest_period_only
                )
                period_text = "latest period" if args.latest_period_only else "all periods"
                print(f"✓ Non-engaged users report ({period_text}) saved to {non_engaged_file}")
                
        except Exception as e:
            print(f"Error in user analysis: {str(e)}")
    
    # Run GPT analysis
    if not args.skip_gpt_trends:
        print("\n=== Running GPT Analysis ===")
        try:
            gpt_analyzer = GPTAnalyzer(args.data_dir, output_dir)
            
            # Generate GPT trend graphs
            gpt_analyzer.generate_all_trends()
            print("✓ GPT trend graphs generated")
            
        except Exception as e:
            print(f"Error in GPT analysis: {str(e)}")
    
    print("\n=== Analysis Complete ===")
    print(f"All results saved to {output_dir}")
    print("Summary of outputs:")
    print("  - User trend graphs: active users, message volume, GPT usage")
    print("  - Message histogram: distribution of average messages per user")
    print("  - GPT trend graphs: active GPTs, GPT messages, unique GPT messagers")
    print("  - User engagement report (CSV)")
    print("  - Non-engaged users report (CSV)")


if __name__ == "__main__":
    main()
