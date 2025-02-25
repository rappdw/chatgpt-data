"""Command-line interface for ChatGPT GPT trends analysis."""

import argparse
import os
from pathlib import Path

from chatgpt_data.analysis.gpt_analysis import GPTAnalysis


def main():
    """Run the GPT trends analysis CLI."""
    parser = argparse.ArgumentParser(
        description="Analyze ChatGPT custom GPT engagement trends"
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
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Run the analysis
    analyzer = GPTAnalysis(args.data_dir, output_dir)
    analyzer.generate_all_trends()

    print(f"GPT trend analysis completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
