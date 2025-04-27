#!/usr/bin/env python
"""
Identify users whose message usage deviates significantly from the average.

This script analyzes user_engagement CSV files and flags users whose message count
exceeds a specified multiplier of the interquartile range (IQR), identifying outliers
that may skew trend visualizations. The IQR method is robust for skewed distributions.
"""

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from chatgpt_data.utils.constants import DEFAULT_TIMEZONE
from chatgpt_data.utils.data_loader import UserDataLoader


class OutlierDetector:
    """Class for detecting outliers in user message usage."""

    def __init__(
        self, data_dir: Union[str, Path], iqr_multiplier: float = 3.0,
        start_date: Optional[str] = None, weekly_chunks: bool = False
    ) -> None:
        """Initialize the OutlierDetector class.

        Args:
            data_dir: Directory containing the user engagement CSV files
            iqr_multiplier: Multiplier of IQR to use as threshold for outlier detection
                           (default: 3.0, where 1.5 indicates mild outliers and 3.0 significant outliers)
            start_date: Optional start date to filter data (format: YYYY-MM-DD)
            weekly_chunks: Whether to process data in weekly chunks
        """
        self.data_dir = Path(data_dir)
        self.iqr_multiplier = iqr_multiplier
        self.start_date = start_date
        self.weekly_chunks = weekly_chunks
        # Don't load user data automatically as it might be loaded later with specific parameters
        self.data_loader = UserDataLoader(data_dir, load_all=False)
        self.user_data = None
        self.outliers_by_period = {}
        self.overall_outliers = {}
        self.pattern_outliers = {}
        self.power_users = {}

    def load_data(self) -> None:
        """Load user engagement data from CSV files."""
        print(f"Loading user engagement data from {self.data_dir}...")
        self.user_data = self.data_loader.get_user_data()
        if self.user_data is None or len(self.user_data) == 0:
            raise ValueError("No user engagement data found")

        # Ensure messages column is numeric
        if not pd.api.types.is_numeric_dtype(self.user_data["messages"]):
            print("Converting messages column to numeric...")
            self.user_data["messages"] = pd.to_numeric(
                self.user_data["messages"], errors="coerce"
            )
            self.user_data = self.user_data.dropna(subset=["messages"])

        # Ensure period_start is datetime
        self.user_data["period_start"] = pd.to_datetime(
            self.user_data["period_start"], utc=True
        ).dt.tz_convert(DEFAULT_TIMEZONE)
        
        # Filter by start date if provided
        if self.start_date:
            start_date = pd.to_datetime(self.start_date).tz_localize(DEFAULT_TIMEZONE)
            print(f"Filtering data from {self.start_date} onwards...")
            self.user_data = self.user_data[self.user_data["period_start"] >= start_date]
            if len(self.user_data) == 0:
                raise ValueError(f"No data found after {self.start_date}")

        print(f"Loaded {len(self.user_data)} user engagement records")

    def detect_outliers_by_period(self) -> Dict[str, pd.DataFrame]:
        """Detect outliers in each time period using the IQR (interquartile range) method, which is robust to skewed distributions.

        Returns:
            Dictionary mapping period start dates to DataFrames of outliers
        """
        if self.user_data is None:
            self.load_data()

        periods = self.user_data["period_start"].unique()
        print(f"Analyzing {len(periods)} time periods for outliers...")

        for period in tqdm(periods):
            period_data = self.user_data[self.user_data["period_start"] == period]

            # Calculate statistics using IQR (robust to skew)
            q1 = period_data["messages"].quantile(0.25)
            q3 = period_data["messages"].quantile(0.75)
            iqr = q3 - q1
            threshold = q3 + self.iqr_multiplier * iqr  # Only upper outliers

            # Identify outliers
            outliers = period_data[period_data["messages"] > threshold].copy()

            if len(outliers) > 0:
                # Add metrics appropriate for skewed distributions
                outliers["period_median"] = period_data["messages"].median()
                outliers["period_q1"] = q1
                outliers["period_q3"] = q3
                outliers["period_iqr"] = iqr
                outliers["threshold"] = threshold
                # When IQR is 0, use the raw distance from Q3 instead of IQR-normalized distance
                outliers["iqr_distance"] = (outliers["messages"] - q3) / iqr if iqr > 0 else (outliers["messages"] - q3)

                # Ensure all required columns exist
                if "email" not in outliers.columns and "user_email" in outliers.columns:
                    outliers["email"] = outliers["user_email"]

                # Format period for dictionary key
                period_str = pd.Timestamp(period).strftime("%Y-%m-%d")
                self.outliers_by_period[period_str] = outliers.sort_values(
                    "messages", ascending=False
                )

        print(f"Found outliers in {len(self.outliers_by_period)} periods")
        return self.outliers_by_period

    def detect_overall_outliers(self) -> pd.DataFrame:
        """Detect users who are consistently outliers across multiple periods.

        Returns:
            DataFrame of users with their outlier frequency and IQR-based metrics
        """
        if not self.outliers_by_period:
            self.detect_outliers_by_period()

        # Combine all period outliers
        all_outliers = pd.concat(
            [df for df in self.outliers_by_period.values()], ignore_index=True
        )

        # Group by user and calculate summary statistics
        overall_outliers = (
            all_outliers.groupby("email")
            .agg(
                {
                    "messages": ["count", "mean", "max"],
                    "iqr_distance": "mean",
                    "period_q1": "mean",
                    "period_q3": "mean",
                    "period_iqr": "mean",
                }
            )
            .round(2)
        )

        # Flatten column names
        overall_outliers.columns = [
            "_".join(col).strip() for col in overall_outliers.columns.values
        ]

        # Rename columns for clarity
        overall_outliers = overall_outliers.rename(
            columns={
                "messages_count": "outlier_periods",
                "messages_mean": "avg_messages",
                "messages_max": "max_messages",
                "iqr_distance_mean": "avg_iqr_distance",
                "period_q1_mean": "avg_period_q1",
                "period_q3_mean": "avg_period_q3",
                "period_iqr_mean": "avg_period_iqr",
            }
        )

        # Sort by frequency of being an outlier, then by average IQR distance
        overall_outliers = overall_outliers.sort_values(
            ["outlier_periods", "avg_iqr_distance"], ascending=[False, False]
        )

        return overall_outliers


    def detect_pattern_outliers(self) -> pd.DataFrame:
        """Detect users whose usage pattern shows significant variation compared to their own average.
        
        This identifies users who have unusual spikes or drops in their own usage pattern,
        rather than comparing to the population average.
        
        Returns:
            DataFrame of users with significant personal usage variations
        """
        if self.user_data is None:
            self.load_data()
            
        # Group by user and calculate statistics for each user's usage pattern
        user_cols = ["email"] if "email" in self.user_data.columns else ["user_email"]
        user_message_stats = (
            self.user_data.groupby(user_cols + ["period_start"])["messages"]
            .sum()
            .reset_index()
        )
        
        # Calculate each user's mean and standard deviation across all periods
        user_pattern_stats = user_message_stats.groupby(user_cols).agg(
            mean_messages=pd.NamedAgg(column="messages", aggfunc="mean"),
            std_messages=pd.NamedAgg(column="messages", aggfunc="std"),
            max_messages=pd.NamedAgg(column="messages", aggfunc="max"),
            min_messages=pd.NamedAgg(column="messages", aggfunc="min"),
            period_count=pd.NamedAgg(column="period_start", aggfunc="count"),
        ).reset_index()
        
        # Replace NaN std_messages (users with only one period) with 0
        user_pattern_stats["std_messages"] = user_pattern_stats["std_messages"].fillna(0)
        
        # Calculate coefficient of variation (std/mean) to identify users with high variability
        # This normalizes the std deviation to account for different usage levels
        user_pattern_stats["coefficient_of_variation"] = (
            user_pattern_stats["std_messages"] / user_pattern_stats["mean_messages"]
        ).fillna(0)
        
        # Calculate max deviation from personal mean (in number of personal std devs)
        user_pattern_stats["max_personal_deviation"] = np.where(
            user_pattern_stats["std_messages"] > 0,
            (user_pattern_stats["max_messages"] - user_pattern_stats["mean_messages"]) / 
            user_pattern_stats["std_messages"],
            0
        )
        
        # Flag users with high personal pattern variability
        # Users must have data for at least 3 periods to be considered for pattern analysis
        pattern_outliers = user_pattern_stats[
            (user_pattern_stats["period_count"] >= 3) & 
            (user_pattern_stats["coefficient_of_variation"] > 0.5) &
            # Use a fixed threshold of 2.0 for personal deviation since this is based on z-scores
            # and not related to the IQR method used for population outliers
            (user_pattern_stats["max_personal_deviation"] > 2.0)
        ].copy()
        
        return pattern_outliers.sort_values("max_personal_deviation", ascending=False)
    
    def identify_power_users(self) -> pd.DataFrame:
        """Identify power users who consistently use more messages than the population average.
        
        Power users are defined as those who:
        1. Have above-average message usage in most periods they appear in
        2. Appear in multiple periods (at least 3)
        3. Have significantly higher average message count than the population average
        
        Returns:
            DataFrame of power users with usage statistics
        """
        if self.user_data is None:
            self.load_data()
            
        # Calculate population-wide message statistics
        population_mean = self.user_data["messages"].mean()
        
        # Group by user and period to get message counts per user per period
        user_cols = ["email"] if "email" in self.user_data.columns else ["user_email"]
        user_period_messages = (
            self.user_data.groupby(user_cols + ["period_start"])["messages"]
            .sum()
            .reset_index()
        )
        
        # For each user and period, check if their usage is above population average
        user_period_messages["above_avg"] = user_period_messages["messages"] > population_mean
        
        # Aggregate user statistics
        user_stats = user_period_messages.groupby(user_cols).agg(
            total_messages=pd.NamedAgg(column="messages", aggfunc="sum"),
            avg_messages=pd.NamedAgg(column="messages", aggfunc="mean"),
            periods_above_avg=pd.NamedAgg(column="above_avg", aggfunc="sum"),
            period_count=pd.NamedAgg(column="period_start", aggfunc="count"),
        ).reset_index()
        
        # Calculate percentage of periods where user was above average
        user_stats["pct_periods_above_avg"] = (
            user_stats["periods_above_avg"] / user_stats["period_count"] * 100
        )
        
        # Calculate how many times higher than population average their usage is
        user_stats["times_above_population_avg"] = user_stats["avg_messages"] / population_mean
        
        # Identify power users: appear in 3+ periods, above avg in 75%+ of periods,
        # and at least 2x the population average usage
        power_users = user_stats[
            (user_stats["period_count"] >= 3) &
            (user_stats["pct_periods_above_avg"] >= 75) &
            (user_stats["times_above_population_avg"] >= 2)
        ].copy()
        
        return power_users.sort_values("times_above_population_avg", ascending=False)

    def detect_pattern_outliers(self) -> pd.DataFrame:
        """Detect users whose usage pattern shows significant variation compared to their own average.
        
        This identifies users who have unusual spikes or drops in their own usage pattern,
        rather than comparing to the population average.
        
        Returns:
            DataFrame of users with significant personal usage variations
        """
        if self.user_data is None:
            self.load_data()
            
        # Group by user and calculate statistics for each user's usage pattern
        user_cols = ["email"] if "email" in self.user_data.columns else ["user_email"]
        user_message_stats = (
            self.user_data.groupby(user_cols + ["period_start"])["messages"]
            .sum()
            .reset_index()
        )
    
        # Calculate each user's mean and standard deviation across all periods
        user_pattern_stats = user_message_stats.groupby(user_cols).agg(
            mean_messages=pd.NamedAgg(column="messages", aggfunc="mean"),
            std_messages=pd.NamedAgg(column="messages", aggfunc="std"),
            max_messages=pd.NamedAgg(column="messages", aggfunc="max"),
            min_messages=pd.NamedAgg(column="messages", aggfunc="min"),
            period_count=pd.NamedAgg(column="period_start", aggfunc="count"),
        ).reset_index()
        
        # Replace NaN std_messages (users with only one period) with 0
        user_pattern_stats["std_messages"] = user_pattern_stats["std_messages"].fillna(0)
        
        # Calculate coefficient of variation (std/mean) to identify users with high variability
        # This normalizes the std deviation to account for different usage levels
        user_pattern_stats["coefficient_of_variation"] = (
            user_pattern_stats["std_messages"] / user_pattern_stats["mean_messages"]
        ).fillna(0)
        
        # Calculate max deviation from personal mean (in number of personal std devs)
        user_pattern_stats["max_personal_deviation"] = np.where(
            user_pattern_stats["std_messages"] > 0,
            (user_pattern_stats["max_messages"] - user_pattern_stats["mean_messages"]) / 
            user_pattern_stats["std_messages"],
            0
        )
        
        # Flag users with high personal pattern variability
        # Users must have data for at least 3 periods to be considered for pattern analysis
        pattern_outliers = user_pattern_stats[
            (user_pattern_stats["period_count"] >= 3) & 
            (user_pattern_stats["coefficient_of_variation"] > 0.5) &
            # Use a fixed threshold of 2.0 for personal deviation since this is based on z-scores
            # and not related to the IQR method used for population outliers
            (user_pattern_stats["max_personal_deviation"] > 2.0)
        ].copy()
        
        return pattern_outliers.sort_values("max_personal_deviation", ascending=False)
    
    def identify_power_users(self) -> pd.DataFrame:
        """Identify power users who consistently use more messages than the population average.
        
        Power users are defined as those who:
        1. Have above-average message usage in most periods they appear in
        2. Appear in multiple periods (at least 3)
        3. Have significantly higher average message count than the population average
        
        Returns:
            DataFrame of power users with usage statistics
        """
        if self.user_data is None:
            self.load_data()
            
        # Calculate population-wide message statistics
        population_mean = self.user_data["messages"].mean()
        
        # Group by user and period to get message counts per user per period
        user_cols = ["email"] if "email" in self.user_data.columns else ["user_email"]
        user_period_messages = (
            self.user_data.groupby(user_cols + ["period_start"])["messages"]
            .sum()
            .reset_index()
        )
        
        # For each user and period, check if their usage is above population average
        user_period_messages["above_avg"] = user_period_messages["messages"] > population_mean
        
        # Aggregate user statistics
        user_stats = user_period_messages.groupby(user_cols).agg(
            total_messages=pd.NamedAgg(column="messages", aggfunc="sum"),
            avg_messages=pd.NamedAgg(column="messages", aggfunc="mean"),
            periods_above_avg=pd.NamedAgg(column="above_avg", aggfunc="sum"),
            period_count=pd.NamedAgg(column="period_start", aggfunc="count"),
        ).reset_index()
        
        # Calculate percentage of periods where user was above average
        user_stats["pct_periods_above_avg"] = (
            user_stats["periods_above_avg"] / user_stats["period_count"] * 100
        )
        
        # Calculate how many times higher than population average their usage is
        user_stats["times_above_population_avg"] = user_stats["avg_messages"] / population_mean
    
        # Identify power users: appear in 3+ periods, above avg in 75%+ of periods,
        # and at least 2x the population average usage
        power_users = user_stats[
            (user_stats["period_count"] >= 3) &
            (user_stats["pct_periods_above_avg"] >= 75) &
            (user_stats["times_above_population_avg"] >= 2)
        ].copy()
        
        return power_users.sort_values("times_above_population_avg", ascending=False)
    
    def save_outlier_report(self, output_dir: Union[str, Path]) -> List[str]:
        """Save outlier reports to CSV files.

        Args:
            output_dir: Directory to save the output CSV files

        Returns:
            List of paths to the generated report files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Ensure we have detected outliers
        if not self.outliers_by_period:
            self.detect_outliers_by_period()

        # Get overall outliers (population-based)
        overall_outliers = self.detect_overall_outliers()
        
        # Get pattern outliers (self-comparison based)
        pattern_outliers = self.detect_pattern_outliers()
        
        # Get power users
        power_users = self.identify_power_users()

        # Save period-by-period outliers
        all_period_outliers = pd.concat(
            [
                df.assign(period=period)
                for period, df in self.outliers_by_period.items()
            ],
            ignore_index=True,
        )
        
        # Add display names to all reports using the refactored method from UserDataLoader
        all_period_outliers = self.data_loader.add_display_names_to_dataframe(all_period_outliers)
        overall_outliers = self.data_loader.add_display_names_to_dataframe(overall_outliers)
        pattern_outliers = self.data_loader.add_display_names_to_dataframe(pattern_outliers)
        power_users = self.data_loader.add_display_names_to_dataframe(power_users)

        # Determine available columns for the report
        base_cols = [
            "period",
            "messages",
            "period_median",
            "period_q1",
            "period_q3",
            "period_iqr",
            "iqr_distance",
            "threshold",
        ]
        
        # Add user identifier columns if they exist
        period_cols = base_cols.copy()
        if "email" in all_period_outliers.columns:
            period_cols.insert(1, "email")
        if "user_email" in all_period_outliers.columns:
            period_cols.insert(1, "user_email")
        if "display_name" in all_period_outliers.columns:
            period_cols.insert(1, "display_name")
            
        # Create report paths
        period_report_path = output_dir / "message_outliers_by_period.csv"
        overall_report_path = output_dir / "message_outliers_overall.csv"
        pattern_report_path = output_dir / "message_pattern_outliers.csv"
        power_users_path = output_dir / "power_users.csv"
        
        # Save all reports
        # 1. Period-by-period outliers
        existing_cols = [col for col in period_cols if col in all_period_outliers.columns]
        all_period_outliers[existing_cols].to_csv(period_report_path, index=False)

        # 2. Overall population-based outliers
        overall_outliers.to_csv(overall_report_path, index=False)
        
        # 3. Pattern outliers (users with high personal variation)
        pattern_outliers.to_csv(pattern_report_path, index=False)
        
        # 4. Power users - reorder columns to have display_name first
        if 'display_name' in power_users.columns:
            # Get the email column name (could be 'email' or 'user_email')
            email_col = 'email' if 'email' in power_users.columns else 'user_email'
            
            # Reorder columns to have display_name first, then email, then the rest
            cols = ['display_name', email_col]
            other_cols = [col for col in power_users.columns if col not in cols]
            power_users = power_users[cols + other_cols]
            
        power_users.to_csv(power_users_path, index=False)

        print(f"Saved period-by-period outlier report to {period_report_path}")
        print(f"Saved overall population-based outlier report to {overall_report_path}")
        print(f"Saved pattern outlier report (personal variation) to {pattern_report_path}")
        print(f"Saved power users report to {power_users_path}")

        return [str(period_report_path), str(overall_report_path), 
                str(pattern_report_path), str(power_users_path)]


def main() -> None:
    """Main entry point for the outlier detection script."""
    parser = argparse.ArgumentParser(
        description="Identify power users and usage pattern outliers"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./rawdata",
        help="Directory containing user engagement CSV files (default: ./rawdata)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reports",
        help="Directory to save outlier reports (default: ./reports)",
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=3.0,
        help="Multiplier of IQR to use as threshold for outlier detection (default: 3.0, where 1.5 indicates mild outliers and 3.0 significant outliers)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date to filter data (format: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--weekly-chunks",
        action="store_true",
        help="Process data in weekly chunks",
    )
    args = parser.parse_args()

    try:
        # Create and run outlier detector
        detector = OutlierDetector(
            data_dir=args.data_dir,
            iqr_multiplier=args.iqr_multiplier,
            start_date=args.start_date,
            weekly_chunks=args.weekly_chunks,
        )
        detector.load_data()
        reports = detector.save_outlier_report(args.output_dir)

        print("\nUser Analysis completed successfully")
        print(f"Generated {len(reports)} reports in {args.output_dir} directory:")
        print("  - Period-by-period outliers (compared to population)")
        print("  - Overall population-based outliers")
        print("  - Pattern outliers (significant variation in personal usage)")
        print("  - Power users (consistently high usage compared to population)")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
