"""User engagement analysis module."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class UserAnalysis:
    """Class for analyzing ChatGPT user engagement data."""

    def __init__(self, data_dir: Union[str, Path], output_dir: Union[str, Path]):
        """Initialize the UserAnalysis class.

        Args:
            data_dir: Directory containing the raw data files
            output_dir: Directory to save the output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.user_data = None
        self._load_data()

    def _load_data(self) -> None:
        """Load all user engagement data files."""
        user_files = [f for f in os.listdir(self.data_dir) if f.startswith("proofpoint_user_engagement")]
        
        dfs = []
        for file in user_files:
            try:
                # Try different encodings if UTF-8 fails
                encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(self.data_dir / file, encoding=encoding)
                        print(f"Successfully loaded {file} with encoding {encoding}")
                        dfs.append(df)
                        break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
        
        if dfs:
            self.user_data = pd.concat(dfs, ignore_index=True)
        else:
            raise FileNotFoundError("No user engagement data files found or could not be loaded")

    def generate_active_users_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of active users over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")

        # Group by period and count active users
        active_users = (
            self.user_data[self.user_data["is_active"] == 1]
            .groupby(["period_start"])
            .size()
            .reset_index(name="active_users")
        )
        
        # Convert period_start to datetime
        active_users["period_start"] = pd.to_datetime(active_users["period_start"])
        
        # Sort by date
        active_users = active_users.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(active_users["period_start"], active_users["active_users"], marker="o")
        ax.set_title("Active Users Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of Active Users")
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "active_users_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_message_volume_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of message volume over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")

        # Group by period and sum messages
        message_volume = (
            self.user_data
            .groupby(["period_start"])
            ["messages"]
            .sum()
            .reset_index()
        )
        
        # Convert period_start to datetime
        message_volume["period_start"] = pd.to_datetime(message_volume["period_start"])
        
        # Sort by date
        message_volume = message_volume.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(message_volume["period_start"], message_volume["messages"], marker="o")
        ax.set_title("Message Volume Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of Messages")
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "message_volume_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_gpt_usage_trend(self, save: bool = True) -> Optional[plt.Figure]:
        """Generate a trend graph of GPT usage over time.

        Args:
            save: Whether to save the figure to the output directory

        Returns:
            The matplotlib figure if save is False, otherwise None
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")

        # Group by period and sum GPT messages
        gpt_usage = (
            self.user_data
            .groupby(["period_start"])
            ["gpt_messages"]
            .sum()
            .reset_index()
        )
        
        # Convert period_start to datetime
        gpt_usage["period_start"] = pd.to_datetime(gpt_usage["period_start"])
        
        # Sort by date
        gpt_usage = gpt_usage.sort_values("period_start")
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(gpt_usage["period_start"], gpt_usage["gpt_messages"], marker="o")
        ax.set_title("GPT Usage Trend")
        ax.set_xlabel("Period")
        ax.set_ylabel("Number of GPT Messages")
        ax.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / "gpt_usage_trend.png"
            plt.savefig(output_path)
            plt.close(fig)
            return None
        
        return fig

    def generate_all_trends(self) -> None:
        """Generate all trend graphs and save them to the output directory."""
        self.generate_active_users_trend()
        self.generate_message_volume_trend()
        self.generate_gpt_usage_trend()
        
    def get_engagement_levels(self, high_threshold: int = 20, low_threshold: int = 5) -> pd.DataFrame:
        """Categorize users by engagement level based on average message count across all periods.
        
        Args:
            high_threshold: Minimum average number of messages to be considered highly engaged
            low_threshold: Maximum average number of messages to be considered low engaged
            
        Returns:
            DataFrame with user information and engagement level
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Filter to only include rows with valid message data
        valid_data = self.user_data[pd.notna(self.user_data["messages"])].copy()
        
        # Calculate average messages per user across all periods
        engagement_df = (
            valid_data
            .groupby(["account_id", "public_id", "name", "email"])
            .agg({
                "messages": "mean",
                "period_start": "min",
                "period_end": "max"
            })
            .reset_index()
        )
        
        # Rename columns for clarity
        engagement_df = engagement_df.rename(columns={
            "messages": "avg_messages",
            "period_start": "first_period",
            "period_end": "last_period"
        })
        
        # Add engagement level column
        conditions = [
            (engagement_df["avg_messages"] >= high_threshold),
            (engagement_df["avg_messages"] <= low_threshold) & (engagement_df["avg_messages"] > 0),
            (engagement_df["avg_messages"] == 0)
        ]
        choices = ["high", "low", "none"]
        engagement_df["engagement_level"] = np.select(conditions, choices, default="medium")
        
        return engagement_df
    
    def generate_engagement_report(self, output_file: str = None) -> pd.DataFrame:
        """Generate a report of user engagement levels based on average message count.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            
        Returns:
            DataFrame with user engagement report
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        # Get the latest period data to identify active users
        latest_period = self.user_data["period_end"].max()
        latest_data = self.user_data[self.user_data["period_end"] == latest_period]
        
        # Filter out users with pending status
        active_users = latest_data[latest_data["user_status"] != "pending"]
        active_user_ids = set(active_users["public_id"].unique())
        
        # Filter to only include rows with valid message data and active users
        valid_data = self.user_data[
            (pd.notna(self.user_data["messages"])) & 
            (self.user_data["public_id"].isin(active_user_ids))
        ].copy()
        
        # Calculate average messages per user across all periods
        user_avg = (
            valid_data
            .groupby(["account_id", "public_id", "name", "email"])
            .agg({
                "messages": "mean",
                "period_start": "min",
                "period_end": "max",
                "is_active": "sum"
            })
            .reset_index()
        )
        
        # Rename columns for clarity
        user_avg = user_avg.rename(columns={
            "messages": "avg_messages",
            "period_start": "first_period",
            "period_end": "last_period",
            "is_active": "active_periods"
        })
        
        # Add engagement level column based on average message count
        conditions = [
            (user_avg["avg_messages"] >= 20),
            (user_avg["avg_messages"] <= 5) & (user_avg["avg_messages"] > 0),
            (user_avg["avg_messages"] == 0)
        ]
        choices = ["high", "low", "none"]
        user_avg["engagement_level"] = np.select(conditions, choices, default="medium")
        
        # Create a custom sort order for engagement levels
        engagement_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
        user_avg["engagement_sort"] = user_avg["engagement_level"].map(engagement_order)
        
        # Sort by engagement level (high to low) and then by average messages (high to low)
        user_avg = user_avg.sort_values(
            by=["engagement_sort", "avg_messages"], 
            ascending=[True, False]
        ).drop(columns=["engagement_sort"])
        
        # Count users by engagement level
        engagement_counts = user_avg["engagement_level"].value_counts().to_dict()
        print(f"Engagement Levels Summary (based on average message count):")
        print(f"  High: {engagement_counts.get('high', 0)}")
        print(f"  Medium: {engagement_counts.get('medium', 0)}")
        print(f"  Low: {engagement_counts.get('low', 0)}")
        print(f"  None: {engagement_counts.get('none', 0)}")
        
        # Save to file if specified
        if output_file:
            user_avg.to_csv(output_file, index=False)
            print(f"Engagement report saved to {output_file}")
            
        return user_avg

    def get_non_engaged_users(self, only_latest_period: bool = False) -> pd.DataFrame:
        """Identify users who have never engaged across all tracked periods or just the latest period.
        
        Args:
            only_latest_period: If True, only consider the latest period for non-engagement
                                If False, identify users who have never engaged across all periods
        
        Returns:
            DataFrame with non-engaged user information
        """
        if self.user_data is None:
            raise ValueError("User data not loaded")
            
        if only_latest_period:
            # Get the latest period data
            latest_period = self.user_data["period_end"].max()
            latest_data = self.user_data[self.user_data["period_end"] == latest_period]
            
            # Filter out pending users
            active_latest_users = latest_data[latest_data["user_status"] != "pending"]
            
            # Find users who were not active in the latest period
            non_engaged_latest = active_latest_users[
                (active_latest_users["is_active"] == 0) | 
                (pd.isna(active_latest_users["is_active"]))
            ]
            
            # Make sure we have all necessary columns
            result = non_engaged_latest.copy()
            
            return result
        else:
            # Get all unique users
            all_users = self.user_data[["account_id", "public_id", "name", "email"]].drop_duplicates()
            
            # Get users who have been active at least once
            active_users = self.user_data[self.user_data["is_active"] == 1][
                ["account_id", "public_id", "name", "email"]
            ].drop_duplicates()
            
            # Find users who have never been active
            never_active = pd.merge(
                all_users, active_users, 
                on=["account_id", "public_id", "email"], 
                how="left", 
                indicator=True,
                suffixes=("", "_active")
            )
            never_active = never_active[never_active["_merge"] == "left_only"].drop(["_merge", "name_active"], axis=1, errors="ignore")
            
            # Get additional user information
            user_info = self.user_data[["account_id", "public_id", "email", "user_role", "role", "department", "user_status", "created_or_invited_date"]].drop_duplicates()
            never_active = pd.merge(
                never_active,
                user_info,
                on=["account_id", "public_id", "email"],
                how="left"
            )
            
            return never_active

    def generate_non_engagement_report(self, output_file: str = None, only_latest_period: bool = False) -> pd.DataFrame:
        """Generate a report of users who have never engaged.
        
        Args:
            output_file: Path to save the report CSV file (optional)
            only_latest_period: If True, only consider the latest period for non-engagement
                                If False, identify users who have never engaged across all periods
            
        Returns:
            DataFrame with non-engaged user report
        """
        # Get non-engaged users
        non_engaged = self.get_non_engaged_users(only_latest_period=only_latest_period)
        
        period_text = "in the latest period" if only_latest_period else "across all tracked periods"
        print(f"Non-Engaged Users Summary {period_text}:")
        print(f"  Total non-engaged users: {len(non_engaged)}")
        
        # Count by user status if the column exists
        if 'user_status' in non_engaged.columns:
            status_counts = non_engaged["user_status"].value_counts().to_dict()
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
        
        # Save to file if specified
        if output_file:
            non_engaged.to_csv(output_file, index=False)
            print(f"Non-engagement report saved to {output_file}")
            
        return non_engaged
